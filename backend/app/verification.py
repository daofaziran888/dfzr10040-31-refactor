"""统一核验凭据与跨检查交接协调。

质量门禁与跨表校验原先各自装配检查报告证据，本模块把两者共有的
「输入标识 + 校验值」收拢为同一份核验凭据结构（VerificationCredential），
并通过凭据编码驱动两件事：

1. 快照发布登记（SnapshotReleaseRegistry）：同一份凭据的重领、重投只做
   去重登记，真实快照写入在回执放开时且仅发生一次；
2. 漂移调度队列（DriftDispatchQueue）：只领取本次交接已确认可处理的
   版本，更正凭据证明原差异消失后撤销待办，但保留原差异供追查。

两端在交接时都凭同一份凭据复核检查类型、输入版本、共享输入身份与校验值。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

# 检查类型常量：门禁与跨表校验共用同一套凭据结构，靠 check_kind 区分。
CHECK_KIND_QUALITY_GATE = "quality_gate"
CHECK_KIND_CROSS_VALIDATE = "cross_validate"

# 发布与漂移待办的状态字面量，集中维护避免两端拼写漂移。
RELEASE_PENDING = "待回执"
RELEASE_PUBLISHED = "已发布"

DRIFT_AWAITING_HANDOFF = "待交接确认"
DRIFT_READY = "待调度"
DRIFT_CLAIMED = "已领取"
DRIFT_REVOKED = "已撤销"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_digest(payload: Any) -> str:
    """对任意可 JSON 化的数据计算稳定校验值，作为宽表版本的指纹。"""
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return "SHA256-" + hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass
class VerificationCredential:
    """统一核验凭据：门禁报告与跨表校验报告共用的证据结构。

    共有输入标识（wide_table / input_version / shared_input_id）与校验值
    （check_value）是两端交接复核的锚点，report 保留各自原有的检查明细，
    门禁侧仍是 passed/violations，跨表侧仍是 mismatch_count/mismatches。
    """

    credential_code: str
    check_kind: str
    wide_table: str
    input_version: str
    shared_input_id: str
    check_value: str
    row_count: int
    report: dict[str, Any]
    status: str
    created_at: str

    def handoff_identity(self) -> dict[str, Any]:
        """交接身份四要素：检查类型、输入版本、共享输入身份、校验值。"""
        return {
            "check_kind": self.check_kind,
            "wide_table": self.wide_table,
            "input_version": self.input_version,
            "shared_input_id": self.shared_input_id,
            "check_value": self.check_value,
        }


def build_verification_credential(
    *,
    check_kind: str,
    wide_table: str,
    input_version: str,
    shared_input_id: str,
    records: list[dict[str, Any]],
    report: dict[str, Any],
    status: str,
    sequence: int,
) -> VerificationCredential:
    """统一证据装配入口：门禁与跨表校验都走这里产出凭据。

    校验值同时覆盖共享输入身份、输入版本、被检数据与检查报告，
    因此同一宽表的不同版本必然得到不同凭据，无法互相冒充证据。
    """
    check_value = _canonical_digest({
        "shared_input_id": shared_input_id,
        "input_version": input_version,
        "wide_table": wide_table,
        "records": records,
        "report": report,
    })
    return VerificationCredential(
        credential_code=f"VCRED-{sequence:04d}",
        check_kind=check_kind,
        wide_table=wide_table,
        input_version=input_version,
        shared_input_id=shared_input_id,
        check_value=check_value,
        row_count=len(records),
        report=dict(report),
        status=status,
        created_at=_now_iso(),
    )


@dataclass
class SnapshotReleaseEntry:
    """一次快照发布请求的登记记录，按凭据编码做幂等。"""

    release_code: str
    credential_code: str
    wide_table: str
    partition_key_value: str
    record_count: int
    status: str = RELEASE_PENDING
    snapshot_code: str = ""
    published_at: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def is_published(self) -> bool:
        return self.status == RELEASE_PUBLISHED


class SnapshotReleaseRegistry:
    """快照发布入口：凭据级幂等，保证同一份凭据只产生一次真实写入。

    作业超时重领（reclaim）与旧凭据被另一进程重投（redeliver）都只追加
    尝试记录，不触发快照写入；只有回执放开（confirm_receipt）时才调用
    注入的 snapshot_writer 完成一次真实写入，之后任何重复请求都被拒绝。
    """

    def __init__(self, snapshot_writer: Callable[[SnapshotReleaseEntry], str]) -> None:
        self._entries: dict[str, SnapshotReleaseEntry] = {}
        self._snapshot_writer = snapshot_writer
        self._next_seq = 1

    def _record_attempt(self, entry: SnapshotReleaseEntry, action: str,
                        worker: str, accepted: bool, reason: str) -> None:
        entry.attempts.append({
            "action": action,
            "worker": worker,
            "at": _now_iso(),
            "accepted": accepted,
            "reason": reason,
        })

    def request_release(self, credential: VerificationCredential,
                        partition_key_value: str,
                        worker: str = "worker-0") -> tuple[SnapshotReleaseEntry, bool, str]:
        """登记发布请求。同一凭据重复到达时不去写快照，只返回原登记。"""
        existing = self._entries.get(credential.credential_code)
        if existing is not None:
            self._record_attempt(existing, "request", worker, False, "凭据已登记，重复请求去重")
            return existing, False, "duplicate_request"

        entry = SnapshotReleaseEntry(
            release_code=f"REL-{self._next_seq:04d}",
            credential_code=credential.credential_code,
            wide_table=credential.wide_table,
            partition_key_value=partition_key_value,
            record_count=credential.row_count,
        )
        self._next_seq += 1
        self._record_attempt(entry, "request", worker, True, "发布请求已受理，等待回执")
        self._entries[credential.credential_code] = entry
        return entry, True, "accepted"

    def reclaim(self, credential_code: str,
                worker: str) -> tuple[SnapshotReleaseEntry | None, bool, str]:
        """作业超时后重新领取：凭据仍有效，但不允许再次写出结果。"""
        entry = self._entries.get(credential_code)
        if entry is None:
            return None, False, "unknown_credential"
        if entry.is_published():
            self._record_attempt(entry, "reclaim", worker, False, "快照已发布，拒绝重复写出")
            return entry, False, "already_published"
        self._record_attempt(entry, "reclaim", worker, False, "发布回执未返回，重领已去重挂起")
        return entry, False, "dedup_pending"

    def redeliver(self, credential_code: str,
                  worker: str) -> tuple[SnapshotReleaseEntry | None, bool, str]:
        """另一工作进程重投旧凭据：同样只登记尝试，不产生第二次写入。"""
        entry = self._entries.get(credential_code)
        if entry is None:
            return None, False, "unknown_credential"
        if entry.is_published():
            self._record_attempt(entry, "redeliver", worker, False, "快照已发布，重投被幂等拦截")
            return entry, False, "already_published"
        self._record_attempt(entry, "redeliver", worker, False, "旧凭据重投已去重，暂缓动作保持挂起")
        return entry, False, "dedup_pending"

    def confirm_receipt(self, credential_code: str,
                        worker: str = "publisher") -> tuple[SnapshotReleaseEntry | None, bool, str]:
        """放开回执：此时且仅此时执行一次真实快照写入。"""
        entry = self._entries.get(credential_code)
        if entry is None:
            return None, False, "unknown_credential"
        if entry.is_published():
            self._record_attempt(entry, "confirm", worker, False, "回执已放开过，拒绝重复发布")
            return entry, False, "already_published"

        snapshot_code = self._snapshot_writer(entry)
        entry.snapshot_code = snapshot_code
        entry.status = RELEASE_PUBLISHED
        entry.published_at = _now_iso()
        self._record_attempt(entry, "confirm", worker, True, "回执放开，快照真实写入一次")
        return entry, True, "published"

    def get_release(self, credential_code: str) -> SnapshotReleaseEntry | None:
        return self._entries.get(credential_code)

    def all_entries(self) -> list[SnapshotReleaseEntry]:
        return list(self._entries.values())


@dataclass
class DriftTodo:
    """漂移处置待办：绑定一份跨表校验凭据及其原始差异证据。"""

    todo_code: str
    credential_code: str
    wide_table: str
    input_version: str
    shared_input_id: str
    check_value: str
    original_report: dict[str, Any]
    status: str = DRIFT_AWAITING_HANDOFF
    handoff_confirmed: bool = False
    handling_records: list[dict[str, Any]] = field(default_factory=list)
    revoked_by_credential: str = ""
    revoked_at: str = ""

    def is_open(self) -> bool:
        return self.status != DRIFT_REVOKED


class DriftDispatchQueue:
    """漂移调度队列：只领取交接已确认可处理的版本。

    更正凭据（同一共享输入身份的新版本、差异计数归零）到达时撤销待办，
    原始差异报告原样保留在 original_report 中供追查。
    """

    def __init__(self) -> None:
        self._todos: dict[str, DriftTodo] = {}
        self._next_seq = 1

    def handoff(self, credential: VerificationCredential) -> DriftTodo:
        """依据跨表校验凭据发起交接，初始状态必须等待交接确认。"""
        if credential.check_kind != CHECK_KIND_CROSS_VALIDATE:
            raise ValueError("只有交叉校验凭据可以发起漂移交接")
        existing = self._todos.get(credential.credential_code)
        if existing is not None:
            return existing
        todo = DriftTodo(
            todo_code=f"DRIFT-{self._next_seq:04d}",
            credential_code=credential.credential_code,
            wide_table=credential.wide_table,
            input_version=credential.input_version,
            shared_input_id=credential.shared_input_id,
            check_value=credential.check_value,
            original_report=dict(credential.report),
        )
        todo.handling_records.append({
            "action": "handoff", "at": _now_iso(),
            "detail": "漂移处置交接已发起，等待独立可处理确认",
        })
        self._next_seq += 1
        self._todos[credential.credential_code] = todo
        return todo

    def confirm_handoff(self, credential_code: str) -> DriftTodo | None:
        """确认该版本可独立发起漂移处置，之后才允许调度领取。"""
        todo = self._todos.get(credential_code)
        if todo is None:
            return None
        todo.handoff_confirmed = True
        if todo.is_open():
            todo.status = DRIFT_READY
        todo.handling_records.append({
            "action": "confirm_handoff", "at": _now_iso(),
            "detail": "交接确认完成，版本进入可调度集合",
        })
        return todo

    def claim_dispatch(self, worker: str = "dispatcher-0") -> DriftTodo | None:
        """领取一个待调度待办；未确认交接或已撤销的版本不会被领取。"""
        for todo in self._todos.values():
            if todo.handoff_confirmed and todo.status == DRIFT_READY:
                todo.status = DRIFT_CLAIMED
                todo.handling_records.append({
                    "action": "claim", "at": _now_iso(), "worker": worker,
                    "detail": f"漂移调度领取 {todo.input_version} 的处置待办",
                })
                return todo
        return None

    def submit_correction(self, new_credential: VerificationCredential
                          ) -> tuple[DriftTodo | None, bool, str]:
        """提交更正比对凭据：差异消失则撤销待办，原差异保留供追查。"""
        if new_credential.check_kind != CHECK_KIND_CROSS_VALIDATE:
            return None, False, "not_cross_validation_credential"

        open_todos = [
            todo for todo in self._todos.values()
            if todo.is_open()
            and todo.shared_input_id == new_credential.shared_input_id
            and todo.wide_table == new_credential.wide_table
        ]
        if not open_todos:
            return None, False, "no_open_todo"
        # 同一共享输入只可能有一个未撤销的漂移待办。
        todo = open_todos[0]

        if int(new_credential.report.get("mismatch_count", 0)) > 0:
            todo.handling_records.append({
                "action": "correction_rejected", "at": _now_iso(),
                "credential_code": new_credential.credential_code,
                "detail": "更正比对仍有差异，待办保持有效",
            })
            return todo, False, "mismatch_remains"

        todo.status = DRIFT_REVOKED
        todo.handoff_confirmed = True
        todo.revoked_by_credential = new_credential.credential_code
        todo.revoked_at = _now_iso()
        todo.handling_records.append({
            "action": "revoke", "at": _now_iso(),
            "credential_code": new_credential.credential_code,
            "detail": "更正凭据证明原差异已消失，撤销待办并保留原差异证据",
        })
        return todo, True, "revoked"

    def get_todo(self, credential_code: str) -> DriftTodo | None:
        return self._todos.get(credential_code)

    def all_todos(self) -> list[DriftTodo]:
        return list(self._todos.values())
