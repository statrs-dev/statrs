#! /bin/bash

process_file() {
  # Define input and output file names
  SOURCE=$1
  FILENAME=$2
  curl -fsSL ${SOURCE}/$FILENAME > $FILENAME

  # Extract line numbers for Certified Values and Data from the header
  INFO=$(grep "Certified Values:" $FILENAME)
  CERTIFIED_VALUES_START=$(echo $INFO | awk '{print $4}')
  CERTIFIED_VALUES_END=$(echo $INFO | awk '{print $6}')

  INFO=$(grep "Data            :" $FILENAME)
  DATA_START=$(echo $INFO | awk '{print $4}')
  DATA_END=$(echo $INFO | awk '{print $6}')

  # Extract and reformat sections
  # Certified values
  sed -n -i \
      -e "${CERTIFIED_VALUES_START},${CERTIFIED_VALUES_END}p" \
      -e "${DATA_START},${DATA_END}p" \
      $FILENAME
  # sed -n -i -e "${CERTIFIED_VALUES_START},${CERTIFIED_VALUES_END}s/\(exact\)//p" $FILENAME

}

URL='https://www.itl.nist.gov/div898/strd/univ/data'
for file in Lottery.dat Lew.dat Mavro.dat Michelso.dat NumAcc1.dat NumAcc2.dat NumAcc3.dat
do
  process_file $URL $file
done

