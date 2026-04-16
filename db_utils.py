"""
SQLite bootstrapper for the healthcare booking agent.

Seeds eight fictional doctors across five specialties and creates an
appointments table. All queries used by the graph go through the helpers
in this module so the booking_graph nodes stay focused on orchestration.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

DB_PATH = os.environ.get("BOOKING_DB_PATH", "booking.db")


SEED_DOCTORS: List[Dict[str, Any]] = [
    {"name": "Dr. Aanya Rao", "specialty": "general_medicine", "years_experience": 9},
    {"name": "Dr. Marcus Feld", "specialty": "general_medicine", "years_experience": 14},
    {"name": "Dr. Priya Natarajan", "specialty": "cardiology", "years_experience": 18},
    {"name": "Dr. Jonah Vega", "specialty": "cardiology", "years_experience": 11},
    {"name": "Dr. Sara Okafor", "specialty": "dermatology", "years_experience": 7},
    {"name": "Dr. Ben Halvorsen", "specialty": "pediatrics", "years_experience": 12},
    {"name": "Dr. Lin Mei", "specialty": "pediatrics", "years_experience": 6},
    {"name": "Dr. Rafael Castillo", "specialty": "orthopedics", "years_experience": 16},
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and seed doctors on first run."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                specialty TEXT NOT NULL,
                years_experience INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                doctor_id INTEGER NOT NULL,
                doctor_name TEXT NOT NULL,
                specialty TEXT NOT NULL,
                slot TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'confirmed',
                created_at TEXT NOT NULL,
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            )
            """
        )

        cur.execute("SELECT COUNT(*) AS n FROM doctors")
        if cur.fetchone()["n"] == 0:
            for d in SEED_DOCTORS:
                cur.execute(
                    "INSERT INTO doctors (name, specialty, years_experience) VALUES (?, ?, ?)",
                    (d["name"], d["specialty"], d["years_experience"]),
                )
        conn.commit()
    finally:
        conn.close()


def list_all_doctors() -> List[Dict[str, Any]]:
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, name, specialty, years_experience FROM doctors ORDER BY specialty, name"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_doctors_by_specialty(specialty: str) -> List[Dict[str, Any]]:
    """Return doctors that match a specialty label, e.g. 'cardiology'."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, name, specialty, years_experience FROM doctors WHERE specialty = ? ORDER BY years_experience DESC",
            (specialty,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_doctor(doctor_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, name, specialty, years_experience FROM doctors WHERE id = ?",
            (doctor_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_slots(doctor_id: int, count: int = 3) -> List[str]:
    """
    Generate deterministic 'available' slots for a given doctor.

    In a real system this would query a calendar service. For this workshop
    we derive three upcoming weekday slots from today so the UX feels real
    without needing a calendar backend.
    """
    base = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(days=1)
    hours = [9, 11, 14, 16]
    slots: List[str] = []
    d = base
    while len(slots) < count:
        if d.weekday() < 5:  # Mon-Fri
            hour = hours[(doctor_id + len(slots)) % len(hours)]
            when = d.replace(hour=hour)
            slots.append(when.strftime("%A %b %d, %I:%M %p"))
        d += timedelta(days=1)
    return slots


def create_appointment(
    patient_id: str,
    doctor_id: int,
    doctor_name: str,
    specialty: str,
    slot: str,
) -> Dict[str, Any]:
    init_db()
    conn = _connect()
    try:
        now = datetime.utcnow().isoformat()
        cur = conn.execute(
            """
            INSERT INTO appointments
                (patient_id, doctor_id, doctor_name, specialty, slot, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'confirmed', ?)
            """,
            (patient_id, doctor_id, doctor_name, specialty, slot, now),
        )
        conn.commit()
        appt_id = cur.lastrowid
        return {
            "id": appt_id,
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "doctor_name": doctor_name,
            "specialty": specialty,
            "slot": slot,
            "status": "confirmed",
            "created_at": now,
        }
    finally:
        conn.close()


def list_appointments_for_patient(patient_id: str) -> List[Dict[str, Any]]:
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, patient_id, doctor_id, doctor_name, specialty, slot, status, created_at
            FROM appointments
            WHERE patient_id = ?
            ORDER BY created_at DESC
            """,
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
