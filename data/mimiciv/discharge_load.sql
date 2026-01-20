USE mimic4;

DROP TABLE IF EXISTS discharge;
CREATE TABLE discharge (
  note_id VARCHAR(25) NOT NULL,
  subject_id INT NOT NULL,
  hadm_id INT NOT NULL,
  note_type VARCHAR(2) NOT NULL,
  note_seq SMALLINT NOT NULL,
  charttime DATETIME NOT NULL,
  storetime DATETIME,
  text TEXT NOT NULL
) CHARACTER SET = UTF8MB4;

LOAD DATA LOCAL INFILE 'discharge.csv' INTO TABLE discharge
  FIELDS TERMINATED BY ',' ESCAPED BY '' OPTIONALLY ENCLOSED BY '"'
  LINES TERMINATED BY '\n'
  IGNORE 1 LINES
  (@note_id,@subject_id,@hadm_id,@note_type,@note_seq,@charttime,@storetime,@text)
SET
  note_id = TRIM(@note_id),
  subject_id = TRIM(@subject_id),
  hadm_id = TRIM(@hadm_id),
  note_type = TRIM(@note_type),
  note_seq = TRIM(@note_seq),
  charttime = TRIM(@charttime),
  storetime = NULLIF(TRIM(@storetime), ''),
  text = @text;

CREATE INDEX discharge_hadm_id_idx ON discharge (hadm_id);
CREATE INDEX discharge_subject_id_idx ON discharge (subject_id);

SHOW WARNINGS;
