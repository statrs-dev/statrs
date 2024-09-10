#! /bin/bash
# this script is to download and preprocess datafiles for the nist_tests.rs
# integration test for statrs downloads data to directory specified by env
# var STATRS_NIST_DATA_DIR

process_file() {
  # Define input and output file names
  SOURCE=$1
  FILENAME=$2
  TARGET=${STATRS_NIST_DATA_DIR-tests}/${FILENAME}
  echo -e ${FILENAME} '\n\tDownloading...'
  curl -fsSL ${SOURCE}/$FILENAME > ${TARGET}

  # Extract line numbers for Certified Values and Data from the header
  INFO=$(grep "Certified Values:" $TARGET)
  CERTIFIED_VALUES_START=$(echo $INFO | awk '{print $4}')
  CERTIFIED_VALUES_END=$(echo $INFO | awk '{print $6}')

  INFO=$(grep "Data            :" $TARGET)
  DATA_START=$(echo $INFO | awk '{print $4}')
  DATA_END=$(echo $INFO | awk '{print $6}')

  echo -e '\tFormatting...'
  # Extract and reformat sections
  sed -n -i \
      -e "${CERTIFIED_VALUES_START},${CERTIFIED_VALUES_END}p" \
      -e "${DATA_START},${DATA_END}p" \
      $TARGET
}

URL='https://www.itl.nist.gov/div898/strd/univ/data'
for file in Lottery.dat Lew.dat Mavro.dat Michelso.dat NumAcc1.dat NumAcc2.dat NumAcc3.dat
do
  process_file $URL $file
done

