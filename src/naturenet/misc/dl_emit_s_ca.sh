#!/bin/bash

GREP_OPTIONS=''

cookiejar=$(mktemp cookies.XXXXXXXXXX)
netrc=$(mktemp netrc.XXXXXXXXXX)
chmod 0600 "$cookiejar" "$netrc"
function finish {
  rm -rf "$cookiejar" "$netrc"
}

trap finish EXIT
WGETRC="$wgetrc"

prompt_credentials() {
    echo "Enter your Earthdata Login or other provider supplied credentials"
    read -p "Username (nlahaye): " username
    username=${username:-nlahaye}
    read -s -p "Password: " password
    echo "machine urs.earthdata.nasa.gov login $username password $password" >> $netrc
    echo
}

exit_with_error() {
    echo
    echo "Unable to Retrieve Data"
    echo
    echo $1
    echo
    echo "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250130T185936_2503013_002/EMIT_L2A_RFL_001_20250130T185936_2503013_002.nc"
    echo
    exit 1
}

prompt_credentials
  detect_app_approval() {
    approved=`curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250130T185936_2503013_002/EMIT_L2A_RFL_001_20250130T185936_2503013_002.nc -w '\n%{http_code}' | tail  -1`
    if [ "$approved" -ne "200" ] && [ "$approved" -ne "301" ] && [ "$approved" -ne "302" ]; then
        # User didn't approve the app. Direct users to approve the app in URS
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below "
    fi
}

setup_auth_curl() {
    # Firstly, check if it require URS authentication
    status=$(curl -s -z "$(date)" -w '\n%{http_code}' https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250130T185936_2503013_002/EMIT_L2A_RFL_001_20250130T185936_2503013_002.nc | tail -1)
    if [[ "$status" -ne "200" && "$status" -ne "304" ]]; then
        # URS authentication is required. Now further check if the application/remote service is approved.
        detect_app_approval
    fi
}

setup_auth_wget() {
    # The safest way to auth via curl is netrc. Note: there's no checking or feedback
    # if login is unsuccessful
    touch ~/.netrc
    chmod 0600 ~/.netrc
    credentials=$(grep 'machine urs.earthdata.nasa.gov' ~/.netrc)
    if [ -z "$credentials" ]; then
        cat "$netrc" >> ~/.netrc
    fi
}

fetch_urls() {
  if command -v curl >/dev/null 2>&1; then
      setup_auth_curl
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        curl -f -b "$cookiejar" -c "$cookiejar" -L --netrc-file "$netrc" -g -o $stripped_query_params -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  elif command -v wget >/dev/null 2>&1; then
      # We can't use wget to poke provider server to get info whether or not URS was integrated without download at least one of the files.
      echo
      echo "WARNING: Can't find curl, use wget instead."
      echo "WARNING: Script may not correctly identify Earthdata Login integrations."
      echo
      setup_auth_wget
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        wget --load-cookies "$cookiejar" --save-cookies "$cookiejar" --output-document $stripped_query_params --keep-session-cookies -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  else
      exit_with_error "Error: Could not find a command-line downloader.  Please install curl or wget"
  fi
}

fetch_urls <<'EDSCEOF'
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250130T185936_2503013_002/EMIT_L2A_RFL_001_20250130T185936_2503013_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250130T185936_2503013_002/EMIT_L2A_RFLUNCERT_001_20250130T185936_2503013_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250130T185936_2503013_002/EMIT_L2A_MASK_001_20250130T185936_2503013_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250219T190344_2505013_001/EMIT_L2A_RFL_001_20250219T190344_2505013_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250219T190344_2505013_001/EMIT_L2A_RFLUNCERT_001_20250219T190344_2505013_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250219T190344_2505013_001/EMIT_L2A_MASK_001_20250219T190344_2505013_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250328T203209_2508713_001/EMIT_L2A_RFL_001_20250328T203209_2508713_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250328T203209_2508713_001/EMIT_L2A_RFLUNCERT_001_20250328T203209_2508713_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250328T203209_2508713_001/EMIT_L2A_MASK_001_20250328T203209_2508713_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250328T203221_2508713_002/EMIT_L2A_RFL_001_20250328T203221_2508713_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250328T203221_2508713_002/EMIT_L2A_RFLUNCERT_001_20250328T203221_2508713_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250328T203221_2508713_002/EMIT_L2A_MASK_001_20250328T203221_2508713_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250328T203233_2508713_003/EMIT_L2A_RFL_001_20250328T203233_2508713_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250328T203233_2508713_003/EMIT_L2A_RFLUNCERT_001_20250328T203233_2508713_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250328T203233_2508713_003/EMIT_L2A_MASK_001_20250328T203233_2508713_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250331T212034_2509014_001/EMIT_L2A_RFL_001_20250331T212034_2509014_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250331T212034_2509014_001/EMIT_L2A_RFLUNCERT_001_20250331T212034_2509014_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250331T212034_2509014_001/EMIT_L2A_MASK_001_20250331T212034_2509014_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250331T212046_2509014_002/EMIT_L2A_RFL_001_20250331T212046_2509014_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250331T212046_2509014_002/EMIT_L2A_RFLUNCERT_001_20250331T212046_2509014_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250331T212046_2509014_002/EMIT_L2A_MASK_001_20250331T212046_2509014_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250531T195856_2515113_002/EMIT_L2A_RFL_001_20250531T195856_2515113_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250531T195856_2515113_002/EMIT_L2A_RFLUNCERT_001_20250531T195856_2515113_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250531T195856_2515113_002/EMIT_L2A_MASK_001_20250531T195856_2515113_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250601T204831_2515214_003/EMIT_L2A_RFL_001_20250601T204831_2515214_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250601T204831_2515214_003/EMIT_L2A_RFLUNCERT_001_20250601T204831_2515214_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250601T204831_2515214_003/EMIT_L2A_MASK_001_20250601T204831_2515214_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250602T195946_2515313_003/EMIT_L2A_RFL_001_20250602T195946_2515313_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250602T195946_2515313_003/EMIT_L2A_RFLUNCERT_001_20250602T195946_2515313_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250602T195946_2515313_003/EMIT_L2A_MASK_001_20250602T195946_2515313_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250603T191103_2515413_004/EMIT_L2A_RFL_001_20250603T191103_2515413_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250603T191103_2515413_004/EMIT_L2A_RFLUNCERT_001_20250603T191103_2515413_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250603T191103_2515413_004/EMIT_L2A_MASK_001_20250603T191103_2515413_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250606T182313_2515712_005/EMIT_L2A_RFL_001_20250606T182313_2515712_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250606T182313_2515712_005/EMIT_L2A_RFLUNCERT_001_20250606T182313_2515712_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250606T182313_2515712_005/EMIT_L2A_MASK_001_20250606T182313_2515712_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250606T231417_2515715_002/EMIT_L2A_RFL_001_20250606T231417_2515715_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250606T231417_2515715_002/EMIT_L2A_RFLUNCERT_001_20250606T231417_2515715_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250606T231417_2515715_002/EMIT_L2A_MASK_001_20250606T231417_2515715_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250606T231429_2515715_003/EMIT_L2A_RFL_001_20250606T231429_2515715_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250606T231429_2515715_003/EMIT_L2A_RFLUNCERT_001_20250606T231429_2515715_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250606T231429_2515715_003/EMIT_L2A_MASK_001_20250606T231429_2515715_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250609T173446_2516011_004/EMIT_L2A_RFL_001_20250609T173446_2516011_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250609T173446_2516011_004/EMIT_L2A_RFLUNCERT_001_20250609T173446_2516011_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250609T173446_2516011_004/EMIT_L2A_MASK_001_20250609T173446_2516011_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250609T222552_2516014_002/EMIT_L2A_RFL_001_20250609T222552_2516014_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250609T222552_2516014_002/EMIT_L2A_RFLUNCERT_001_20250609T222552_2516014_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250609T222552_2516014_002/EMIT_L2A_MASK_001_20250609T222552_2516014_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250609T222604_2516014_003/EMIT_L2A_RFL_001_20250609T222604_2516014_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250609T222604_2516014_003/EMIT_L2A_RFLUNCERT_001_20250609T222604_2516014_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250609T222604_2516014_003/EMIT_L2A_MASK_001_20250609T222604_2516014_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250611T173437_2516211_002/EMIT_L2A_RFL_001_20250611T173437_2516211_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250611T173437_2516211_002/EMIT_L2A_RFLUNCERT_001_20250611T173437_2516211_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250611T173437_2516211_002/EMIT_L2A_MASK_001_20250611T173437_2516211_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250615T204916_2516613_004/EMIT_L2A_RFL_001_20250615T204916_2516613_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250615T204916_2516613_004/EMIT_L2A_RFLUNCERT_001_20250615T204916_2516613_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250615T204916_2516613_004/EMIT_L2A_MASK_001_20250615T204916_2516613_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250616T200007_2516713_002/EMIT_L2A_RFL_001_20250616T200007_2516713_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250616T200007_2516713_002/EMIT_L2A_RFLUNCERT_001_20250616T200007_2516713_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250616T200007_2516713_002/EMIT_L2A_MASK_001_20250616T200007_2516713_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250618T200204_2516913_003/EMIT_L2A_RFL_001_20250618T200204_2516913_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250618T200204_2516913_003/EMIT_L2A_RFLUNCERT_001_20250618T200204_2516913_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250618T200204_2516913_003/EMIT_L2A_MASK_001_20250618T200204_2516913_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250619T191122_2517012_002/EMIT_L2A_RFL_001_20250619T191122_2517012_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250619T191122_2517012_002/EMIT_L2A_RFLUNCERT_001_20250619T191122_2517012_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250619T191122_2517012_002/EMIT_L2A_MASK_001_20250619T191122_2517012_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182357_2517312_002/EMIT_L2A_RFL_001_20250622T182357_2517312_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182357_2517312_002/EMIT_L2A_RFLUNCERT_001_20250622T182357_2517312_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182357_2517312_002/EMIT_L2A_MASK_001_20250622T182357_2517312_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182409_2517312_003/EMIT_L2A_RFL_001_20250622T182409_2517312_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182409_2517312_003/EMIT_L2A_RFLUNCERT_001_20250622T182409_2517312_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182409_2517312_003/EMIT_L2A_MASK_001_20250622T182409_2517312_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182421_2517312_004/EMIT_L2A_RFL_001_20250622T182421_2517312_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182421_2517312_004/EMIT_L2A_RFLUNCERT_001_20250622T182421_2517312_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182421_2517312_004/EMIT_L2A_MASK_001_20250622T182421_2517312_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182433_2517312_005/EMIT_L2A_RFL_001_20250622T182433_2517312_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182433_2517312_005/EMIT_L2A_RFLUNCERT_001_20250622T182433_2517312_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182433_2517312_005/EMIT_L2A_MASK_001_20250622T182433_2517312_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182445_2517312_006/EMIT_L2A_RFL_001_20250622T182445_2517312_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182445_2517312_006/EMIT_L2A_RFLUNCERT_001_20250622T182445_2517312_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182445_2517312_006/EMIT_L2A_MASK_001_20250622T182445_2517312_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182456_2517312_007/EMIT_L2A_RFL_001_20250622T182456_2517312_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182456_2517312_007/EMIT_L2A_RFLUNCERT_001_20250622T182456_2517312_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250622T182456_2517312_007/EMIT_L2A_MASK_001_20250622T182456_2517312_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250630T151117_2518110_002/EMIT_L2A_RFL_001_20250630T151117_2518110_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250630T151117_2518110_002/EMIT_L2A_RFLUNCERT_001_20250630T151117_2518110_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250630T151117_2518110_002/EMIT_L2A_MASK_001_20250630T151117_2518110_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250630T151129_2518110_003/EMIT_L2A_RFL_001_20250630T151129_2518110_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250630T151129_2518110_003/EMIT_L2A_RFLUNCERT_001_20250630T151129_2518110_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250630T151129_2518110_003/EMIT_L2A_MASK_001_20250630T151129_2518110_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250630T151141_2518110_004/EMIT_L2A_RFL_001_20250630T151141_2518110_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250630T151141_2518110_004/EMIT_L2A_RFLUNCERT_001_20250630T151141_2518110_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250630T151141_2518110_004/EMIT_L2A_MASK_001_20250630T151141_2518110_004.nc
EDSCEOF
