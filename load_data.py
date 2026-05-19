import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "teiko.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "cell-count.csv")

SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    subject_id   TEXT PRIMARY KEY,
    project      TEXT NOT NULL,
    condition    TEXT,
    age          INTEGER,
    sex          TEXT,
    treatment    TEXT,
    response     TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id                  TEXT PRIMARY KEY,
    subject_id                 TEXT NOT NULL REFERENCES subjects(subject_id),
    sample_type                TEXT,
    time_from_treatment_start  INTEGER
);

CREATE TABLE IF NOT EXISTS cell_counts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id   TEXT NOT NULL REFERENCES samples(sample_id),
    b_cell      INTEGER,
    cd8_t_cell  INTEGER,
    cd4_t_cell  INTEGER,
    nk_cell     INTEGER,
    monocyte    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_samples_subject ON samples(subject_id);
CREATE INDEX IF NOT EXISTS idx_cell_counts_sample ON cell_counts(sample_id);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("DROP TABLE IF EXISTS cell_counts; DROP TABLE IF EXISTS samples; DROP TABLE IF EXISTS subjects;")
    conn.executescript(SCHEMA)
    conn.commit()


def load_csv(conn: sqlite3.Connection, csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    subjects = (
        df[["subject", "project", "condition", "age", "sex", "treatment", "response"]]
        .drop_duplicates(subset=["subject"])
        .rename(columns={"subject": "subject_id"})
    )
    subjects.to_sql("subjects", conn, if_exists="append", index=False)

    samples = df[["sample", "subject", "sample_type", "time_from_treatment_start"]].rename(
        columns={"sample": "sample_id", "subject": "subject_id"}
    )
    samples.to_sql("samples", conn, if_exists="append", index=False)

    counts = df[["sample", "b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]].rename(
        columns={"sample": "sample_id"}
    )
    counts.to_sql("cell_counts", conn, if_exists="append", index=False)

    conn.commit()
    print(f"Loaded {len(df)} rows: {len(subjects)} subjects, {len(samples)} samples.")


def main() -> None:
    print(f"Initializing database at {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    try:
        init_db(conn)
        load_csv(conn, CSV_PATH)
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
