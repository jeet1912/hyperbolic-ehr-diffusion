USE mimic4;

-- Update these paths if you want a different export location.
-- Note: SELECT ... INTO OUTFILE fails if the file already exists.

SELECT * FROM llemr_mortality_train
INTO OUTFILE '/Users/sv.xxt/Downloads/mimic-iv-3.1/exports/mortality_train.csv'
FIELDS TERMINATED BY ',' ESCAPED BY '' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT * FROM llemr_mortality_val
INTO OUTFILE '/Users/sv.xxt/Downloads/mimic-iv-3.1/exports/mortality_val.csv'
FIELDS TERMINATED BY ',' ESCAPED BY '' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT * FROM llemr_mortality_test
INTO OUTFILE '/Users/sv.xxt/Downloads/mimic-iv-3.1/exports/mortality_test.csv'
FIELDS TERMINATED BY ',' ESCAPED BY '' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT * FROM llemr_los_train
INTO OUTFILE '/Users/sv.xxt/Downloads/mimic-iv-3.1/exports/los_train.csv'
FIELDS TERMINATED BY ',' ESCAPED BY '' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT * FROM llemr_los_val
INTO OUTFILE '/Users/sv.xxt/Downloads/mimic-iv-3.1/exports/los_val.csv'
FIELDS TERMINATED BY ',' ESCAPED BY '' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT * FROM llemr_los_test
INTO OUTFILE '/Users/sv.xxt/Downloads/mimic-iv-3.1/exports/los_test.csv'
FIELDS TERMINATED BY ',' ESCAPED BY '' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT * FROM llemr_readmission_train
INTO OUTFILE '/Users/sv.xxt/Downloads/mimic-iv-3.1/exports/readmission_train.csv'
FIELDS TERMINATED BY ',' ESCAPED BY '' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT * FROM llemr_readmission_val
INTO OUTFILE '/Users/sv.xxt/Downloads/mimic-iv-3.1/exports/readmission_val.csv'
FIELDS TERMINATED BY ',' ESCAPED BY '' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT * FROM llemr_readmission_test
INTO OUTFILE '/Users/sv.xxt/Downloads/mimic-iv-3.1/exports/readmission_test.csv'
FIELDS TERMINATED BY ',' ESCAPED BY '' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n';
