"""统一核验凭据交接剧本测试。

覆盖：门禁与跨表校验共用证据结构、凭据级幂等的快照发布（重领/重投不重复
写入）、漂移调度只领取已确认交接的版本、更正凭据撤销待办但保留原差异。
"""
from __future__ import annotations

import pytest

from app.fusion_engine import FusionEngine

# 同一宽表两个版本共享的输入身份：交接两端都要能核对它。
SHARED_INPUT_ID = "SI-ORDER-FUSION-20261123"
WIDE_TABLE = "WT_FURNACE"


def _rows(amounts: list[int]) -> list[dict]:
    return [
        {"id": f"K{i}", "name": f"订单{i}", "amount": amount}
        for i, amount in enumerate(amounts, start=1)
    ]


@pytest.fixture
def engine_and_ids() -> tuple[FusionEngine, int, int]:
    eng = FusionEngine()
    # 阈值 4 的行数下限门禁。
    gate = eng.create_quality_gate(
        target_wide_table=WIDE_TABLE,
        check_type="row_count_min",
        threshold_condition="4",
    )
    # 跨表校验方案：按 id 对齐，比对金额字段。
    plan = eng.create_cross_validate_plan(
        primary_wide_table=WIDE_TABLE,
        reference_wide_table="WT_REFERENCE",
        join_key="id",
        validate_fields=["amount"],
    )
    return eng, gate.rule_id, plan.plan_id


@pytest.fixture
def engine(engine_and_ids: tuple[FusionEngine, int, int]) -> FusionEngine:
    return engine_and_ids[0]


class TestUnifiedCredentialHandoff:
    """完整交接剧本：门禁阻断/通过 -> 发布挂起 -> 漂移交接 -> 重领重投 -> 回执 -> 更正。"""

    def test_full_handoff_scenario(
        self, engine_and_ids: tuple[FusionEngine, int, int],
    ) -> None:
        engine, gate_rule_id, plan_id = engine_and_ids
        # 1. 3 行 + 阈值 4：门禁必须挡住未达标输入，违例明细保持原样。
        blocked = engine.run_quality_gate_check(
            target_wide_table=WIDE_TABLE,
            input_version="V0-PRECHECK",
            shared_input_id=SHARED_INPUT_ID,
            records=_rows([100, 200, 300]),
            gate_rule_id=gate_rule_id,
        )
        assert blocked["passed"] is False
        assert blocked["gates_checked"] == 1
        assert blocked["violations"] == [{
            "rule_code": blocked["violations"][0]["rule_code"],
            "check": "row_count_min",
            "actual": 3,
            "threshold": "4",
        }]
        blocked_cred = blocked["credential"]
        assert blocked_cred["check_kind"] == "quality_gate"
        assert blocked_cred["input_version"] == "V0-PRECHECK"
        assert blocked_cred["shared_input_id"] == SHARED_INPUT_ID
        assert blocked_cred["status"] == "阻断"

        # 2. V1 用 4 行通过门禁，通过字段与（空）违例明细保持。
        v1_rows = _rows([100, 200, 300, 400])
        v1_result = engine.run_quality_gate_check(
            target_wide_table=WIDE_TABLE,
            input_version="V1",
            shared_input_id=SHARED_INPUT_ID,
            records=v1_rows,
            gate_rule_id=gate_rule_id,
        )
        assert v1_result["passed"] is True
        assert v1_result["violations"] == []
        v1_cred_code = v1_result["credential"]["credential_code"]
        v1_cred = engine.get_credential(v1_cred_code)
        assert v1_cred is not None
        assert v1_cred["check_kind"] == "quality_gate"
        assert v1_cred["input_version"] == "V1"
        assert v1_cred["shared_input_id"] == SHARED_INPUT_ID
        assert v1_cred["row_count"] == 4

        # 3. 快照写入请求交给发布入口，回执未返回前没有真实快照。
        release = engine.request_snapshot_release(
            v1_cred_code, partition_key_value="20261123", worker="worker-v1",
        )
        assert release["ok"] is True
        assert release["created"] is True
        assert release["status"] == "待回执"
        assert engine.list_snapshots() == []

        # 4. V2 跨表比对有一笔金额差异，差异计数保持原有口径。
        reference_rows = _rows([100, 200, 399, 400])  # K3 金额 300 -> 399
        v2_rows = _rows([100, 200, 300, 400])
        v2_result = engine.run_cross_validation_check(
            plan_id=plan_id,
            input_version="V2",
            shared_input_id=SHARED_INPUT_ID,
            primary_records=v2_rows,
            reference_records=reference_rows,
        )
        assert v2_result["ok"] is True
        assert v2_result["mismatch_count"] == 1
        assert v2_result["mismatches"][0]["field"] == "amount"
        assert v2_result["mismatches"][0]["primary_value"] == "300"
        assert v2_result["mismatches"][0]["reference_value"] == "399"
        v2_cred_code = v2_result["credential"]["credential_code"]
        assert v2_result["credential"]["check_kind"] == "cross_validate"
        assert v2_result["credential"]["input_version"] == "V2"
        assert v2_result["credential"]["shared_input_id"] == SHARED_INPUT_ID

        # 同共享输入、不同版本 => 凭据编码与校验值都不同。
        assert v2_cred_code != v1_cred_code
        assert v2_result["credential"]["check_value"] != v1_cred["check_value"]

        # 5. V2 漂移交接：未确认前调度领取不到任何版本。
        handoff = engine.handoff_drift(v2_cred_code)
        assert handoff["status"] == "待交接确认"
        assert handoff["handoff_confirmed"] is False
        assert engine.claim_drift_dispatch()["claimed"] is False

        confirm = engine.confirm_drift_handoff(v2_cred_code)
        assert confirm["status"] == "待调度"
        # 调度只会领取本次交接已确认可处理的版本（即 V2）。
        claim = engine.claim_drift_dispatch(worker="dispatcher-a")
        assert claim["claimed"] is True
        assert claim["input_version"] == "V2"
        assert claim["status"] == "已领取"
        # 已被领取后再次调度没有新待办。
        assert engine.claim_drift_dispatch()["claimed"] is False

        # 6. 回执未到期间，重领与旧凭据重投都到达：动作被暂缓，无真实写入。
        reclaim = engine.reclaim_snapshot_release(v1_cred_code, worker="worker-reclaim")
        assert reclaim["accepted"] is False
        assert reclaim["reason"] == "dedup_pending"
        redeliver = engine.redeliver_snapshot_release(v1_cred_code, worker="worker-other")
        assert redeliver["accepted"] is False
        assert redeliver["reason"] == "dedup_pending"
        assert engine.list_snapshots() == []

        # 7. 放开回执：V1 只发布一次，真实快照记录数为 1 且行数为 4。
        receipt = engine.confirm_snapshot_receipt(v1_cred_code)
        assert receipt["ok"] is True
        assert receipt["status"] == "已发布"
        snapshots = engine.list_snapshots()
        assert len(snapshots) == 1
        snap = snapshots[0]
        assert snap.record_count == 4
        assert snap.target_wide_table == WIDE_TABLE
        assert snap.status_mark == "可用"
        assert snap.partition_code == receipt["snapshot_code"]

        # 发布后的重领、重投、重复回执都不能再次写出结果。
        again_reclaim = engine.reclaim_snapshot_release(v1_cred_code)
        again_redeliver = engine.redeliver_snapshot_release(v1_cred_code)
        again_receipt = engine.confirm_snapshot_receipt(v1_cred_code)
        assert again_reclaim["reason"] == "already_published"
        assert again_redeliver["reason"] == "already_published"
        assert again_receipt["reason"] == "already_published"
        assert len(engine.list_snapshots()) == 1

        # 8. V2 的漂移明细不能成为 V1 的检查证据：两份凭据报告互相独立。
        v1_cred_after = engine.get_credential(v1_cred_code)
        assert "violations" in v1_cred_after["report"]
        assert "mismatch_count" not in v1_cred_after["report"]
        assert "mismatches" not in v1_cred_after["report"]
        v2_cred_after = engine.get_credential(v2_cred_code)
        assert v2_cred_after["report"]["mismatch_count"] == 1
        # 跨表报告不携带门禁字段，两份证据互不串用。
        assert "violations" not in v2_cred_after["report"]
        assert "passed" not in v2_cred_after["report"]

        # 9. 提交 V2 更正比对（新凭据 V3）：原差异消失，待办被新凭据撤销。
        correction = engine.run_cross_validation_check(
            plan_id=plan_id,
            input_version="V3",
            shared_input_id=SHARED_INPUT_ID,
            primary_records=_rows([100, 200, 399, 400]),
            reference_records=reference_rows,
        )
        assert correction["mismatch_count"] == 0
        revoke = engine.submit_correction(correction["credential"]["credential_code"])
        assert revoke["revoked"] is True
        assert revoke["status"] == "已撤销"
        assert revoke["revoked_by_credential"] == correction["credential"]["credential_code"]
        # 原差异保留供追查。
        assert revoke["original_report"]["mismatch_count"] == 1
        assert revoke["original_report"]["mismatches"][0]["field"] == "amount"
        # 撤销后调度队列不再有可领取待办。
        assert engine.claim_drift_dispatch()["claimed"] is False

        # 10. V1 已发布快照仍可读；每次交接都可核对检查类型、版本与共享输入身份。
        readable = engine.list_snapshots()
        assert len(readable) == 1
        assert readable[0].partition_code == snap.partition_code
        for credential_code, kind, version in [
            (blocked_cred["credential_code"], "quality_gate", "V0-PRECHECK"),
            (v1_cred_code, "quality_gate", "V1"),
            (v2_cred_code, "cross_validate", "V2"),
            (correction["credential"]["credential_code"], "cross_validate", "V3"),
        ]:
            view = engine.get_credential(credential_code)
            assert view["check_kind"] == kind
            assert view["input_version"] == version
            assert view["shared_input_id"] == SHARED_INPUT_ID
            assert view["check_value"].startswith("SHA256-")

    def test_correction_with_remaining_mismatch_keeps_todo(
        self, engine_and_ids: tuple[FusionEngine, int, int],
    ) -> None:
        """更正比对仍有差异时不能撤销漂移待办。"""
        engine, _gate_rule_id, plan_id = engine_and_ids
        reference_rows = _rows([100, 200, 399, 400])
        v2 = engine.run_cross_validation_check(
            plan_id=plan_id, input_version="V2", shared_input_id=SHARED_INPUT_ID,
            primary_records=_rows([100, 200, 300, 400]),
            reference_records=reference_rows,
        )
        engine.handoff_drift(v2["credential"]["credential_code"])
        engine.confirm_drift_handoff(v2["credential"]["credential_code"])
        engine.claim_drift_dispatch()

        bad_correction = engine.run_cross_validation_check(
            plan_id=plan_id, input_version="V2-FIX1", shared_input_id=SHARED_INPUT_ID,
            primary_records=_rows([100, 201, 300, 400]),
            reference_records=reference_rows,
        )
        result = engine.submit_correction(bad_correction["credential"]["credential_code"])
        assert result["revoked"] is False
        assert result["reason"] == "mismatch_remains"
        assert result["status"] == "已领取"
        assert result["original_report"]["mismatch_count"] == 1

    def test_release_request_unknown_credential_rejected(
        self, engine_and_ids: tuple[FusionEngine, int, int],
    ) -> None:
        """非门禁凭据不能用于快照发布登记。"""
        engine, _gate_rule_id, plan_id = engine_and_ids
        cross = engine.run_cross_validation_check(
            plan_id=plan_id, input_version="VX", shared_input_id=SHARED_INPUT_ID,
            primary_records=_rows([1, 2, 3, 4]),
            reference_records=_rows([1, 2, 3, 4]),
        )
        result = engine.request_snapshot_release(
            cross["credential"]["credential_code"], partition_key_value="20261123",
        )
        assert result["ok"] is False
        assert engine.list_snapshots() == []
