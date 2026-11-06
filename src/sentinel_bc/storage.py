"""Local anonymized review records. Original model JSON is immutable."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import uuid
import pandas as pd


class ReviewStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL, original_output TEXT NOT NULL,
                    measurements TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING',
                    reviewer_decision TEXT, notes TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL REFERENCES cases(case_id),
                    created_at TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS preserve_model_output
                BEFORE UPDATE OF original_output, model_version, measurements ON cases
                BEGIN SELECT RAISE(ABORT, 'Original model record is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS preserve_audit_update BEFORE UPDATE ON audit
                BEGIN SELECT RAISE(ABORT, 'Audit records are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS preserve_audit_delete BEFORE DELETE ON audit
                BEGIN SELECT RAISE(ABORT, 'Audit records are append-only'); END;
            """)

    @contextmanager
    def connection(self):
        with sqlite3.connect(self.path, timeout=15) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            yield db

    def add(self, output: dict, measurements: dict):
        case_id = "case_" + uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as db:
            db.execute("INSERT INTO cases(case_id,created_at,model_version,original_output,measurements) VALUES(?,?,?,?,?)",
                       (case_id, now, output["model_version"], json.dumps(output, allow_nan=False), json.dumps(measurements, allow_nan=False)))
            db.execute("INSERT INTO audit(case_id,created_at,event,payload) VALUES(?,?,?,?)",
                       (case_id, now, "MODEL_RECORDED", json.dumps({"decision": output["decision"]})))
        return case_id

    def list_cases(self, status="ALL"):
        with self.connection() as db:
            query = "SELECT case_id,created_at,model_version,status,reviewer_decision,notes FROM cases"
            if status != "ALL":
                return pd.read_sql_query(query + " WHERE status=? ORDER BY created_at DESC", db, params=[status])
            return pd.read_sql_query(query + " ORDER BY created_at DESC", db)

    def inspect(self, case_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown case ID")
            result = dict(row)
            result["original_output"] = json.loads(result["original_output"])
            result["measurements"] = json.loads(result["measurements"])
            result["audit"] = [dict(r) for r in db.execute("SELECT * FROM audit WHERE case_id=? ORDER BY id", (case_id,))]
            return result

    def review(self, case_id, decision, notes=""):
        if decision not in ["BENIGN", "MALIGNANT", "FURTHER_REVIEW"]:
            raise ValueError("Unsupported reviewer decision")
        if len(notes) > 2000:
            raise ValueError("Notes must be at most 2000 characters; never enter personal data")
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as db:
            row = db.execute("SELECT reviewer_decision,notes FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown case ID")
            db.execute("UPDATE cases SET status='REVIEWED',reviewer_decision=?,notes=? WHERE case_id=?", (decision, notes, case_id))
            payload = {"previous": dict(row), "decision": decision, "notes": notes}
            db.execute("INSERT INTO audit(case_id,created_at,event,payload) VALUES(?,?,?,?)", (case_id, now, "HUMAN_REVIEW", json.dumps(payload)))

    def export(self, path):
        # Exclude free text, measurements and potentially identifying note content.
        with self.connection() as db:
            frame = pd.read_sql_query("SELECT case_id,model_version,status,reviewer_decision FROM cases ORDER BY created_at", db)
        frame.to_csv(path, index=False)
        return str(path)

