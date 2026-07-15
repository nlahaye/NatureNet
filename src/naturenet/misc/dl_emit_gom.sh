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
    echo "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162646_2521211_014/EMIT_L2A_RFL_001_20250731T162646_2521211_014.nc"
    echo
    exit 1
}

prompt_credentials
  detect_app_approval() {
    approved=`curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162646_2521211_014/EMIT_L2A_RFL_001_20250731T162646_2521211_014.nc -w '\n%{http_code}' | tail  -1`
    if [ "$approved" -ne "200" ] && [ "$approved" -ne "301" ] && [ "$approved" -ne "302" ]; then
        # User didn't approve the app. Direct users to approve the app in URS
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below "
    fi
}

setup_auth_curl() {
    # Firstly, check if it require URS authentication
    status=$(curl -s -z "$(date)" -w '\n%{http_code}' https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162646_2521211_014/EMIT_L2A_RFL_001_20250731T162646_2521211_014.nc | tail -1)
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
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162646_2521211_014/EMIT_L2A_RFL_001_20250731T162646_2521211_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162646_2521211_014/EMIT_L2A_RFLUNCERT_001_20250731T162646_2521211_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162646_2521211_014/EMIT_L2A_MASK_001_20250731T162646_2521211_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162634_2521211_013/EMIT_L2A_RFL_001_20250731T162634_2521211_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162634_2521211_013/EMIT_L2A_RFLUNCERT_001_20250731T162634_2521211_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162634_2521211_013/EMIT_L2A_MASK_001_20250731T162634_2521211_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162337_2521211_012/EMIT_L2A_RFL_001_20250731T162337_2521211_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162337_2521211_012/EMIT_L2A_RFLUNCERT_001_20250731T162337_2521211_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162337_2521211_012/EMIT_L2A_MASK_001_20250731T162337_2521211_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162326_2521211_011/EMIT_L2A_RFL_001_20250731T162326_2521211_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162326_2521211_011/EMIT_L2A_RFLUNCERT_001_20250731T162326_2521211_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162326_2521211_011/EMIT_L2A_MASK_001_20250731T162326_2521211_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162314_2521211_010/EMIT_L2A_RFL_001_20250731T162314_2521211_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162314_2521211_010/EMIT_L2A_RFLUNCERT_001_20250731T162314_2521211_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162314_2521211_010/EMIT_L2A_MASK_001_20250731T162314_2521211_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162302_2521211_009/EMIT_L2A_RFL_001_20250731T162302_2521211_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162302_2521211_009/EMIT_L2A_RFLUNCERT_001_20250731T162302_2521211_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T162302_2521211_009/EMIT_L2A_MASK_001_20250731T162302_2521211_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145041_2521210_014/EMIT_L2A_RFL_001_20250731T145041_2521210_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145041_2521210_014/EMIT_L2A_RFLUNCERT_001_20250731T145041_2521210_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145041_2521210_014/EMIT_L2A_MASK_001_20250731T145041_2521210_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145030_2521210_013/EMIT_L2A_RFL_001_20250731T145030_2521210_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145030_2521210_013/EMIT_L2A_RFLUNCERT_001_20250731T145030_2521210_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145030_2521210_013/EMIT_L2A_MASK_001_20250731T145030_2521210_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145018_2521210_012/EMIT_L2A_RFL_001_20250731T145018_2521210_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145018_2521210_012/EMIT_L2A_RFLUNCERT_001_20250731T145018_2521210_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145018_2521210_012/EMIT_L2A_MASK_001_20250731T145018_2521210_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145006_2521210_011/EMIT_L2A_RFL_001_20250731T145006_2521210_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145006_2521210_011/EMIT_L2A_RFLUNCERT_001_20250731T145006_2521210_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T145006_2521210_011/EMIT_L2A_MASK_001_20250731T145006_2521210_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T144954_2521210_010/EMIT_L2A_RFL_001_20250731T144954_2521210_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T144954_2521210_010/EMIT_L2A_RFLUNCERT_001_20250731T144954_2521210_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250731T144954_2521210_010/EMIT_L2A_MASK_001_20250731T144954_2521210_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171445_2521111_016/EMIT_L2A_RFL_001_20250730T171445_2521111_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171445_2521111_016/EMIT_L2A_RFLUNCERT_001_20250730T171445_2521111_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171445_2521111_016/EMIT_L2A_MASK_001_20250730T171445_2521111_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171433_2521111_015/EMIT_L2A_RFL_001_20250730T171433_2521111_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171433_2521111_015/EMIT_L2A_RFLUNCERT_001_20250730T171433_2521111_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171433_2521111_015/EMIT_L2A_MASK_001_20250730T171433_2521111_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171422_2521111_014/EMIT_L2A_RFL_001_20250730T171422_2521111_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171422_2521111_014/EMIT_L2A_RFLUNCERT_001_20250730T171422_2521111_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171422_2521111_014/EMIT_L2A_MASK_001_20250730T171422_2521111_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171410_2521111_013/EMIT_L2A_RFL_001_20250730T171410_2521111_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171410_2521111_013/EMIT_L2A_RFLUNCERT_001_20250730T171410_2521111_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171410_2521111_013/EMIT_L2A_MASK_001_20250730T171410_2521111_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171346_2521111_011/EMIT_L2A_RFL_001_20250730T171346_2521111_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171346_2521111_011/EMIT_L2A_RFLUNCERT_001_20250730T171346_2521111_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171346_2521111_011/EMIT_L2A_MASK_001_20250730T171346_2521111_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171334_2521111_010/EMIT_L2A_RFL_001_20250730T171334_2521111_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171334_2521111_010/EMIT_L2A_RFLUNCERT_001_20250730T171334_2521111_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171334_2521111_010/EMIT_L2A_MASK_001_20250730T171334_2521111_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171144_2521111_009/EMIT_L2A_RFL_001_20250730T171144_2521111_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171144_2521111_009/EMIT_L2A_RFLUNCERT_001_20250730T171144_2521111_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171144_2521111_009/EMIT_L2A_MASK_001_20250730T171144_2521111_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171132_2521111_008/EMIT_L2A_RFL_001_20250730T171132_2521111_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171132_2521111_008/EMIT_L2A_RFLUNCERT_001_20250730T171132_2521111_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T171132_2521111_008/EMIT_L2A_MASK_001_20250730T171132_2521111_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153906_2521110_012/EMIT_L2A_RFL_001_20250730T153906_2521110_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153906_2521110_012/EMIT_L2A_RFLUNCERT_001_20250730T153906_2521110_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153906_2521110_012/EMIT_L2A_MASK_001_20250730T153906_2521110_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153855_2521110_011/EMIT_L2A_RFL_001_20250730T153855_2521110_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153855_2521110_011/EMIT_L2A_RFLUNCERT_001_20250730T153855_2521110_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153855_2521110_011/EMIT_L2A_MASK_001_20250730T153855_2521110_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153843_2521110_010/EMIT_L2A_RFL_001_20250730T153843_2521110_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153843_2521110_010/EMIT_L2A_RFLUNCERT_001_20250730T153843_2521110_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153843_2521110_010/EMIT_L2A_MASK_001_20250730T153843_2521110_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153831_2521110_009/EMIT_L2A_RFL_001_20250730T153831_2521110_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153831_2521110_009/EMIT_L2A_RFLUNCERT_001_20250730T153831_2521110_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153831_2521110_009/EMIT_L2A_MASK_001_20250730T153831_2521110_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153819_2521110_008/EMIT_L2A_RFL_001_20250730T153819_2521110_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153819_2521110_008/EMIT_L2A_RFLUNCERT_001_20250730T153819_2521110_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153819_2521110_008/EMIT_L2A_MASK_001_20250730T153819_2521110_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153753_2521110_007/EMIT_L2A_RFL_001_20250730T153753_2521110_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153753_2521110_007/EMIT_L2A_RFLUNCERT_001_20250730T153753_2521110_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153753_2521110_007/EMIT_L2A_MASK_001_20250730T153753_2521110_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153741_2521110_006/EMIT_L2A_RFL_001_20250730T153741_2521110_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153741_2521110_006/EMIT_L2A_RFLUNCERT_001_20250730T153741_2521110_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153741_2521110_006/EMIT_L2A_MASK_001_20250730T153741_2521110_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153729_2521110_005/EMIT_L2A_RFL_001_20250730T153729_2521110_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153729_2521110_005/EMIT_L2A_RFLUNCERT_001_20250730T153729_2521110_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250730T153729_2521110_005/EMIT_L2A_MASK_001_20250730T153729_2521110_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T180125_2521012_013/EMIT_L2A_RFL_001_20250729T180125_2521012_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T180125_2521012_013/EMIT_L2A_RFLUNCERT_001_20250729T180125_2521012_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T180125_2521012_013/EMIT_L2A_MASK_001_20250729T180125_2521012_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162723_2521011_020/EMIT_L2A_RFL_001_20250729T162723_2521011_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162723_2521011_020/EMIT_L2A_RFLUNCERT_001_20250729T162723_2521011_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162723_2521011_020/EMIT_L2A_MASK_001_20250729T162723_2521011_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162711_2521011_019/EMIT_L2A_RFL_001_20250729T162711_2521011_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162711_2521011_019/EMIT_L2A_RFLUNCERT_001_20250729T162711_2521011_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162711_2521011_019/EMIT_L2A_MASK_001_20250729T162711_2521011_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162659_2521011_018/EMIT_L2A_RFL_001_20250729T162659_2521011_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162659_2521011_018/EMIT_L2A_RFLUNCERT_001_20250729T162659_2521011_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162659_2521011_018/EMIT_L2A_MASK_001_20250729T162659_2521011_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162647_2521011_017/EMIT_L2A_RFL_001_20250729T162647_2521011_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162647_2521011_017/EMIT_L2A_RFLUNCERT_001_20250729T162647_2521011_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162647_2521011_017/EMIT_L2A_MASK_001_20250729T162647_2521011_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162635_2521011_016/EMIT_L2A_RFL_001_20250729T162635_2521011_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162635_2521011_016/EMIT_L2A_RFLUNCERT_001_20250729T162635_2521011_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162635_2521011_016/EMIT_L2A_MASK_001_20250729T162635_2521011_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162624_2521011_015/EMIT_L2A_RFL_001_20250729T162624_2521011_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162624_2521011_015/EMIT_L2A_RFLUNCERT_001_20250729T162624_2521011_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162624_2521011_015/EMIT_L2A_MASK_001_20250729T162624_2521011_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162612_2521011_014/EMIT_L2A_RFL_001_20250729T162612_2521011_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162612_2521011_014/EMIT_L2A_RFLUNCERT_001_20250729T162612_2521011_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162612_2521011_014/EMIT_L2A_MASK_001_20250729T162612_2521011_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162600_2521011_013/EMIT_L2A_RFL_001_20250729T162600_2521011_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162600_2521011_013/EMIT_L2A_RFLUNCERT_001_20250729T162600_2521011_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162600_2521011_013/EMIT_L2A_MASK_001_20250729T162600_2521011_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162548_2521011_012/EMIT_L2A_RFL_001_20250729T162548_2521011_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162548_2521011_012/EMIT_L2A_RFLUNCERT_001_20250729T162548_2521011_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162548_2521011_012/EMIT_L2A_MASK_001_20250729T162548_2521011_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162536_2521011_011/EMIT_L2A_RFL_001_20250729T162536_2521011_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162536_2521011_011/EMIT_L2A_RFLUNCERT_001_20250729T162536_2521011_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250729T162536_2521011_011/EMIT_L2A_MASK_001_20250729T162536_2521011_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171603_2520911_025/EMIT_L2A_RFL_001_20250728T171603_2520911_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171603_2520911_025/EMIT_L2A_RFLUNCERT_001_20250728T171603_2520911_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171603_2520911_025/EMIT_L2A_MASK_001_20250728T171603_2520911_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171552_2520911_024/EMIT_L2A_RFL_001_20250728T171552_2520911_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171552_2520911_024/EMIT_L2A_RFLUNCERT_001_20250728T171552_2520911_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171552_2520911_024/EMIT_L2A_MASK_001_20250728T171552_2520911_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171528_2520911_022/EMIT_L2A_RFL_001_20250728T171528_2520911_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171528_2520911_022/EMIT_L2A_RFLUNCERT_001_20250728T171528_2520911_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171528_2520911_022/EMIT_L2A_MASK_001_20250728T171528_2520911_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171516_2520911_021/EMIT_L2A_RFL_001_20250728T171516_2520911_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171516_2520911_021/EMIT_L2A_RFLUNCERT_001_20250728T171516_2520911_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171516_2520911_021/EMIT_L2A_MASK_001_20250728T171516_2520911_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171504_2520911_020/EMIT_L2A_RFL_001_20250728T171504_2520911_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171504_2520911_020/EMIT_L2A_RFLUNCERT_001_20250728T171504_2520911_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171504_2520911_020/EMIT_L2A_MASK_001_20250728T171504_2520911_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171452_2520911_019/EMIT_L2A_RFL_001_20250728T171452_2520911_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171452_2520911_019/EMIT_L2A_RFLUNCERT_001_20250728T171452_2520911_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171452_2520911_019/EMIT_L2A_MASK_001_20250728T171452_2520911_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171441_2520911_018/EMIT_L2A_RFL_001_20250728T171441_2520911_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171441_2520911_018/EMIT_L2A_RFLUNCERT_001_20250728T171441_2520911_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171441_2520911_018/EMIT_L2A_MASK_001_20250728T171441_2520911_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171429_2520911_017/EMIT_L2A_RFL_001_20250728T171429_2520911_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171429_2520911_017/EMIT_L2A_RFLUNCERT_001_20250728T171429_2520911_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171429_2520911_017/EMIT_L2A_MASK_001_20250728T171429_2520911_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171417_2520911_016/EMIT_L2A_RFL_001_20250728T171417_2520911_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171417_2520911_016/EMIT_L2A_RFLUNCERT_001_20250728T171417_2520911_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171417_2520911_016/EMIT_L2A_MASK_001_20250728T171417_2520911_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171405_2520911_015/EMIT_L2A_RFL_001_20250728T171405_2520911_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171405_2520911_015/EMIT_L2A_RFLUNCERT_001_20250728T171405_2520911_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171405_2520911_015/EMIT_L2A_MASK_001_20250728T171405_2520911_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171353_2520911_014/EMIT_L2A_RFL_001_20250728T171353_2520911_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171353_2520911_014/EMIT_L2A_RFLUNCERT_001_20250728T171353_2520911_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171353_2520911_014/EMIT_L2A_MASK_001_20250728T171353_2520911_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171341_2520911_013/EMIT_L2A_RFL_001_20250728T171341_2520911_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171341_2520911_013/EMIT_L2A_RFLUNCERT_001_20250728T171341_2520911_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T171341_2520911_013/EMIT_L2A_MASK_001_20250728T171341_2520911_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T154020_2520910_029/EMIT_L2A_RFL_001_20250728T154020_2520910_029.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T154020_2520910_029/EMIT_L2A_RFLUNCERT_001_20250728T154020_2520910_029.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T154020_2520910_029/EMIT_L2A_MASK_001_20250728T154020_2520910_029.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T154008_2520910_028/EMIT_L2A_RFL_001_20250728T154008_2520910_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T154008_2520910_028/EMIT_L2A_RFLUNCERT_001_20250728T154008_2520910_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T154008_2520910_028/EMIT_L2A_MASK_001_20250728T154008_2520910_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T153956_2520910_027/EMIT_L2A_RFL_001_20250728T153956_2520910_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T153956_2520910_027/EMIT_L2A_RFLUNCERT_001_20250728T153956_2520910_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T153956_2520910_027/EMIT_L2A_MASK_001_20250728T153956_2520910_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T153944_2520910_026/EMIT_L2A_RFL_001_20250728T153944_2520910_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T153944_2520910_026/EMIT_L2A_RFLUNCERT_001_20250728T153944_2520910_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250728T153944_2520910_026/EMIT_L2A_MASK_001_20250728T153944_2520910_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180426_2520812_012/EMIT_L2A_RFL_001_20250727T180426_2520812_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180426_2520812_012/EMIT_L2A_RFLUNCERT_001_20250727T180426_2520812_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180426_2520812_012/EMIT_L2A_MASK_001_20250727T180426_2520812_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180414_2520812_011/EMIT_L2A_RFL_001_20250727T180414_2520812_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180414_2520812_011/EMIT_L2A_RFLUNCERT_001_20250727T180414_2520812_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180414_2520812_011/EMIT_L2A_MASK_001_20250727T180414_2520812_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180106_2520812_010/EMIT_L2A_RFL_001_20250727T180106_2520812_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180106_2520812_010/EMIT_L2A_RFLUNCERT_001_20250727T180106_2520812_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180106_2520812_010/EMIT_L2A_MASK_001_20250727T180106_2520812_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180054_2520812_009/EMIT_L2A_RFL_001_20250727T180054_2520812_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180054_2520812_009/EMIT_L2A_RFLUNCERT_001_20250727T180054_2520812_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180054_2520812_009/EMIT_L2A_MASK_001_20250727T180054_2520812_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180042_2520812_008/EMIT_L2A_RFL_001_20250727T180042_2520812_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180042_2520812_008/EMIT_L2A_RFLUNCERT_001_20250727T180042_2520812_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180042_2520812_008/EMIT_L2A_MASK_001_20250727T180042_2520812_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180030_2520812_007/EMIT_L2A_RFL_001_20250727T180030_2520812_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180030_2520812_007/EMIT_L2A_RFLUNCERT_001_20250727T180030_2520812_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180030_2520812_007/EMIT_L2A_MASK_001_20250727T180030_2520812_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180018_2520812_006/EMIT_L2A_RFL_001_20250727T180018_2520812_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180018_2520812_006/EMIT_L2A_RFLUNCERT_001_20250727T180018_2520812_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T180018_2520812_006/EMIT_L2A_MASK_001_20250727T180018_2520812_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162803_2520811_011/EMIT_L2A_RFL_001_20250727T162803_2520811_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162803_2520811_011/EMIT_L2A_RFLUNCERT_001_20250727T162803_2520811_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162803_2520811_011/EMIT_L2A_MASK_001_20250727T162803_2520811_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162751_2520811_010/EMIT_L2A_RFL_001_20250727T162751_2520811_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162751_2520811_010/EMIT_L2A_RFLUNCERT_001_20250727T162751_2520811_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162751_2520811_010/EMIT_L2A_MASK_001_20250727T162751_2520811_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162740_2520811_009/EMIT_L2A_RFL_001_20250727T162740_2520811_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162740_2520811_009/EMIT_L2A_RFLUNCERT_001_20250727T162740_2520811_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162740_2520811_009/EMIT_L2A_MASK_001_20250727T162740_2520811_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162728_2520811_008/EMIT_L2A_RFL_001_20250727T162728_2520811_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162728_2520811_008/EMIT_L2A_RFLUNCERT_001_20250727T162728_2520811_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162728_2520811_008/EMIT_L2A_MASK_001_20250727T162728_2520811_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162716_2520811_007/EMIT_L2A_RFL_001_20250727T162716_2520811_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162716_2520811_007/EMIT_L2A_RFLUNCERT_001_20250727T162716_2520811_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162716_2520811_007/EMIT_L2A_MASK_001_20250727T162716_2520811_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162704_2520811_006/EMIT_L2A_RFL_001_20250727T162704_2520811_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162704_2520811_006/EMIT_L2A_RFLUNCERT_001_20250727T162704_2520811_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250727T162704_2520811_006/EMIT_L2A_MASK_001_20250727T162704_2520811_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185137_2520712_015/EMIT_L2A_RFL_001_20250726T185137_2520712_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185137_2520712_015/EMIT_L2A_RFLUNCERT_001_20250726T185137_2520712_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185137_2520712_015/EMIT_L2A_MASK_001_20250726T185137_2520712_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185125_2520712_014/EMIT_L2A_RFL_001_20250726T185125_2520712_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185125_2520712_014/EMIT_L2A_RFLUNCERT_001_20250726T185125_2520712_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185125_2520712_014/EMIT_L2A_MASK_001_20250726T185125_2520712_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185113_2520712_013/EMIT_L2A_RFL_001_20250726T185113_2520712_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185113_2520712_013/EMIT_L2A_RFLUNCERT_001_20250726T185113_2520712_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185113_2520712_013/EMIT_L2A_MASK_001_20250726T185113_2520712_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185101_2520712_012/EMIT_L2A_RFL_001_20250726T185101_2520712_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185101_2520712_012/EMIT_L2A_RFLUNCERT_001_20250726T185101_2520712_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185101_2520712_012/EMIT_L2A_MASK_001_20250726T185101_2520712_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185049_2520712_011/EMIT_L2A_RFL_001_20250726T185049_2520712_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185049_2520712_011/EMIT_L2A_RFLUNCERT_001_20250726T185049_2520712_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T185049_2520712_011/EMIT_L2A_MASK_001_20250726T185049_2520712_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T184911_2520712_010/EMIT_L2A_RFL_001_20250726T184911_2520712_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T184911_2520712_010/EMIT_L2A_RFLUNCERT_001_20250726T184911_2520712_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T184911_2520712_010/EMIT_L2A_MASK_001_20250726T184911_2520712_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171550_2520711_016/EMIT_L2A_RFL_001_20250726T171550_2520711_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171550_2520711_016/EMIT_L2A_RFLUNCERT_001_20250726T171550_2520711_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171550_2520711_016/EMIT_L2A_MASK_001_20250726T171550_2520711_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171538_2520711_015/EMIT_L2A_RFL_001_20250726T171538_2520711_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171538_2520711_015/EMIT_L2A_RFLUNCERT_001_20250726T171538_2520711_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171538_2520711_015/EMIT_L2A_MASK_001_20250726T171538_2520711_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171526_2520711_014/EMIT_L2A_RFL_001_20250726T171526_2520711_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171526_2520711_014/EMIT_L2A_RFLUNCERT_001_20250726T171526_2520711_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171526_2520711_014/EMIT_L2A_MASK_001_20250726T171526_2520711_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171515_2520711_013/EMIT_L2A_RFL_001_20250726T171515_2520711_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171515_2520711_013/EMIT_L2A_RFLUNCERT_001_20250726T171515_2520711_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171515_2520711_013/EMIT_L2A_MASK_001_20250726T171515_2520711_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171503_2520711_012/EMIT_L2A_RFL_001_20250726T171503_2520711_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171503_2520711_012/EMIT_L2A_RFLUNCERT_001_20250726T171503_2520711_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171503_2520711_012/EMIT_L2A_MASK_001_20250726T171503_2520711_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171451_2520711_011/EMIT_L2A_RFL_001_20250726T171451_2520711_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171451_2520711_011/EMIT_L2A_RFLUNCERT_001_20250726T171451_2520711_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171451_2520711_011/EMIT_L2A_MASK_001_20250726T171451_2520711_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171439_2520711_010/EMIT_L2A_RFL_001_20250726T171439_2520711_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171439_2520711_010/EMIT_L2A_RFLUNCERT_001_20250726T171439_2520711_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250726T171439_2520711_010/EMIT_L2A_MASK_001_20250726T171439_2520711_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185329_2520512_020/EMIT_L2A_RFL_001_20250724T185329_2520512_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185329_2520512_020/EMIT_L2A_RFLUNCERT_001_20250724T185329_2520512_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185329_2520512_020/EMIT_L2A_MASK_001_20250724T185329_2520512_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185317_2520512_019/EMIT_L2A_RFL_001_20250724T185317_2520512_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185317_2520512_019/EMIT_L2A_RFLUNCERT_001_20250724T185317_2520512_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185317_2520512_019/EMIT_L2A_MASK_001_20250724T185317_2520512_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185253_2520512_017/EMIT_L2A_RFL_001_20250724T185253_2520512_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185253_2520512_017/EMIT_L2A_RFLUNCERT_001_20250724T185253_2520512_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185253_2520512_017/EMIT_L2A_MASK_001_20250724T185253_2520512_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185241_2520512_016/EMIT_L2A_RFL_001_20250724T185241_2520512_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185241_2520512_016/EMIT_L2A_RFLUNCERT_001_20250724T185241_2520512_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185241_2520512_016/EMIT_L2A_MASK_001_20250724T185241_2520512_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185229_2520512_015/EMIT_L2A_RFL_001_20250724T185229_2520512_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185229_2520512_015/EMIT_L2A_RFLUNCERT_001_20250724T185229_2520512_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185229_2520512_015/EMIT_L2A_MASK_001_20250724T185229_2520512_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185058_2520512_014/EMIT_L2A_RFL_001_20250724T185058_2520512_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185058_2520512_014/EMIT_L2A_RFLUNCERT_001_20250724T185058_2520512_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T185058_2520512_014/EMIT_L2A_MASK_001_20250724T185058_2520512_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171715_2520511_024/EMIT_L2A_RFL_001_20250724T171715_2520511_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171715_2520511_024/EMIT_L2A_RFLUNCERT_001_20250724T171715_2520511_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171715_2520511_024/EMIT_L2A_MASK_001_20250724T171715_2520511_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171703_2520511_023/EMIT_L2A_RFL_001_20250724T171703_2520511_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171703_2520511_023/EMIT_L2A_RFLUNCERT_001_20250724T171703_2520511_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171703_2520511_023/EMIT_L2A_MASK_001_20250724T171703_2520511_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171651_2520511_022/EMIT_L2A_RFL_001_20250724T171651_2520511_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171651_2520511_022/EMIT_L2A_RFLUNCERT_001_20250724T171651_2520511_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171651_2520511_022/EMIT_L2A_MASK_001_20250724T171651_2520511_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171639_2520511_021/EMIT_L2A_RFL_001_20250724T171639_2520511_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171639_2520511_021/EMIT_L2A_RFLUNCERT_001_20250724T171639_2520511_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250724T171639_2520511_021/EMIT_L2A_MASK_001_20250724T171639_2520511_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180508_2520412_012/EMIT_L2A_RFL_001_20250723T180508_2520412_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180508_2520412_012/EMIT_L2A_RFLUNCERT_001_20250723T180508_2520412_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180508_2520412_012/EMIT_L2A_MASK_001_20250723T180508_2520412_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180456_2520412_011/EMIT_L2A_RFL_001_20250723T180456_2520412_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180456_2520412_011/EMIT_L2A_RFLUNCERT_001_20250723T180456_2520412_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180456_2520412_011/EMIT_L2A_MASK_001_20250723T180456_2520412_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180445_2520412_010/EMIT_L2A_RFL_001_20250723T180445_2520412_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180445_2520412_010/EMIT_L2A_RFLUNCERT_001_20250723T180445_2520412_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180445_2520412_010/EMIT_L2A_MASK_001_20250723T180445_2520412_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180433_2520412_009/EMIT_L2A_RFL_001_20250723T180433_2520412_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180433_2520412_009/EMIT_L2A_RFLUNCERT_001_20250723T180433_2520412_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180433_2520412_009/EMIT_L2A_MASK_001_20250723T180433_2520412_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180421_2520412_008/EMIT_L2A_RFL_001_20250723T180421_2520412_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180421_2520412_008/EMIT_L2A_RFLUNCERT_001_20250723T180421_2520412_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180421_2520412_008/EMIT_L2A_MASK_001_20250723T180421_2520412_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180409_2520412_007/EMIT_L2A_RFL_001_20250723T180409_2520412_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180409_2520412_007/EMIT_L2A_RFLUNCERT_001_20250723T180409_2520412_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250723T180409_2520412_007/EMIT_L2A_MASK_001_20250723T180409_2520412_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202837_2520313_019/EMIT_L2A_RFL_001_20250722T202837_2520313_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202837_2520313_019/EMIT_L2A_RFLUNCERT_001_20250722T202837_2520313_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202837_2520313_019/EMIT_L2A_MASK_001_20250722T202837_2520313_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202825_2520313_018/EMIT_L2A_RFL_001_20250722T202825_2520313_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202825_2520313_018/EMIT_L2A_RFLUNCERT_001_20250722T202825_2520313_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202825_2520313_018/EMIT_L2A_MASK_001_20250722T202825_2520313_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202813_2520313_017/EMIT_L2A_RFL_001_20250722T202813_2520313_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202813_2520313_017/EMIT_L2A_RFLUNCERT_001_20250722T202813_2520313_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202813_2520313_017/EMIT_L2A_MASK_001_20250722T202813_2520313_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202802_2520313_016/EMIT_L2A_RFL_001_20250722T202802_2520313_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202802_2520313_016/EMIT_L2A_RFLUNCERT_001_20250722T202802_2520313_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202802_2520313_016/EMIT_L2A_MASK_001_20250722T202802_2520313_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202750_2520313_015/EMIT_L2A_RFL_001_20250722T202750_2520313_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202750_2520313_015/EMIT_L2A_RFLUNCERT_001_20250722T202750_2520313_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202750_2520313_015/EMIT_L2A_MASK_001_20250722T202750_2520313_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202738_2520313_014/EMIT_L2A_RFL_001_20250722T202738_2520313_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202738_2520313_014/EMIT_L2A_RFLUNCERT_001_20250722T202738_2520313_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202738_2520313_014/EMIT_L2A_MASK_001_20250722T202738_2520313_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202726_2520313_013/EMIT_L2A_RFL_001_20250722T202726_2520313_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202726_2520313_013/EMIT_L2A_RFLUNCERT_001_20250722T202726_2520313_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202726_2520313_013/EMIT_L2A_MASK_001_20250722T202726_2520313_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202714_2520313_012/EMIT_L2A_RFL_001_20250722T202714_2520313_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202714_2520313_012/EMIT_L2A_RFLUNCERT_001_20250722T202714_2520313_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202714_2520313_012/EMIT_L2A_MASK_001_20250722T202714_2520313_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202623_2520313_011/EMIT_L2A_RFL_001_20250722T202623_2520313_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202623_2520313_011/EMIT_L2A_RFLUNCERT_001_20250722T202623_2520313_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T202623_2520313_011/EMIT_L2A_MASK_001_20250722T202623_2520313_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185326_2520312_020/EMIT_L2A_RFL_001_20250722T185326_2520312_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185326_2520312_020/EMIT_L2A_RFLUNCERT_001_20250722T185326_2520312_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185326_2520312_020/EMIT_L2A_MASK_001_20250722T185326_2520312_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185314_2520312_019/EMIT_L2A_RFL_001_20250722T185314_2520312_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185314_2520312_019/EMIT_L2A_RFLUNCERT_001_20250722T185314_2520312_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185314_2520312_019/EMIT_L2A_MASK_001_20250722T185314_2520312_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185302_2520312_018/EMIT_L2A_RFL_001_20250722T185302_2520312_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185302_2520312_018/EMIT_L2A_RFLUNCERT_001_20250722T185302_2520312_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185302_2520312_018/EMIT_L2A_MASK_001_20250722T185302_2520312_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185250_2520312_017/EMIT_L2A_RFL_001_20250722T185250_2520312_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185250_2520312_017/EMIT_L2A_RFLUNCERT_001_20250722T185250_2520312_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185250_2520312_017/EMIT_L2A_MASK_001_20250722T185250_2520312_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185226_2520312_015/EMIT_L2A_RFL_001_20250722T185226_2520312_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185226_2520312_015/EMIT_L2A_RFLUNCERT_001_20250722T185226_2520312_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185226_2520312_015/EMIT_L2A_MASK_001_20250722T185226_2520312_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185215_2520312_014/EMIT_L2A_RFL_001_20250722T185215_2520312_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185215_2520312_014/EMIT_L2A_RFLUNCERT_001_20250722T185215_2520312_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185215_2520312_014/EMIT_L2A_MASK_001_20250722T185215_2520312_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185203_2520312_013/EMIT_L2A_RFL_001_20250722T185203_2520312_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185203_2520312_013/EMIT_L2A_RFLUNCERT_001_20250722T185203_2520312_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185203_2520312_013/EMIT_L2A_MASK_001_20250722T185203_2520312_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185151_2520312_012/EMIT_L2A_RFL_001_20250722T185151_2520312_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185151_2520312_012/EMIT_L2A_RFLUNCERT_001_20250722T185151_2520312_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185151_2520312_012/EMIT_L2A_MASK_001_20250722T185151_2520312_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185139_2520312_011/EMIT_L2A_RFL_001_20250722T185139_2520312_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185139_2520312_011/EMIT_L2A_RFLUNCERT_001_20250722T185139_2520312_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250722T185139_2520312_011/EMIT_L2A_MASK_001_20250722T185139_2520312_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194057_2520213_013/EMIT_L2A_RFL_001_20250721T194057_2520213_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194057_2520213_013/EMIT_L2A_RFLUNCERT_001_20250721T194057_2520213_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194057_2520213_013/EMIT_L2A_MASK_001_20250721T194057_2520213_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194045_2520213_012/EMIT_L2A_RFL_001_20250721T194045_2520213_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194045_2520213_012/EMIT_L2A_RFLUNCERT_001_20250721T194045_2520213_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194045_2520213_012/EMIT_L2A_MASK_001_20250721T194045_2520213_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194033_2520213_011/EMIT_L2A_RFL_001_20250721T194033_2520213_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194033_2520213_011/EMIT_L2A_RFLUNCERT_001_20250721T194033_2520213_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194033_2520213_011/EMIT_L2A_MASK_001_20250721T194033_2520213_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194021_2520213_010/EMIT_L2A_RFL_001_20250721T194021_2520213_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194021_2520213_010/EMIT_L2A_RFLUNCERT_001_20250721T194021_2520213_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194021_2520213_010/EMIT_L2A_MASK_001_20250721T194021_2520213_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194009_2520213_009/EMIT_L2A_RFL_001_20250721T194009_2520213_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194009_2520213_009/EMIT_L2A_RFLUNCERT_001_20250721T194009_2520213_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250721T194009_2520213_009/EMIT_L2A_MASK_001_20250721T194009_2520213_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T203023_2520113_020/EMIT_L2A_RFL_001_20250720T203023_2520113_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T203023_2520113_020/EMIT_L2A_RFLUNCERT_001_20250720T203023_2520113_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T203023_2520113_020/EMIT_L2A_MASK_001_20250720T203023_2520113_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T203012_2520113_019/EMIT_L2A_RFL_001_20250720T203012_2520113_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T203012_2520113_019/EMIT_L2A_RFLUNCERT_001_20250720T203012_2520113_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T203012_2520113_019/EMIT_L2A_MASK_001_20250720T203012_2520113_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202948_2520113_017/EMIT_L2A_RFL_001_20250720T202948_2520113_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202948_2520113_017/EMIT_L2A_RFLUNCERT_001_20250720T202948_2520113_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202948_2520113_017/EMIT_L2A_MASK_001_20250720T202948_2520113_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202936_2520113_016/EMIT_L2A_RFL_001_20250720T202936_2520113_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202936_2520113_016/EMIT_L2A_RFLUNCERT_001_20250720T202936_2520113_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202936_2520113_016/EMIT_L2A_MASK_001_20250720T202936_2520113_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202924_2520113_015/EMIT_L2A_RFL_001_20250720T202924_2520113_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202924_2520113_015/EMIT_L2A_RFLUNCERT_001_20250720T202924_2520113_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202924_2520113_015/EMIT_L2A_MASK_001_20250720T202924_2520113_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202748_2520113_014/EMIT_L2A_RFL_001_20250720T202748_2520113_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202748_2520113_014/EMIT_L2A_RFLUNCERT_001_20250720T202748_2520113_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202748_2520113_014/EMIT_L2A_MASK_001_20250720T202748_2520113_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202736_2520113_013/EMIT_L2A_RFL_001_20250720T202736_2520113_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202736_2520113_013/EMIT_L2A_RFLUNCERT_001_20250720T202736_2520113_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T202736_2520113_013/EMIT_L2A_MASK_001_20250720T202736_2520113_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T185348_2520112_028/EMIT_L2A_RFL_001_20250720T185348_2520112_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T185348_2520112_028/EMIT_L2A_RFLUNCERT_001_20250720T185348_2520112_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T185348_2520112_028/EMIT_L2A_MASK_001_20250720T185348_2520112_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T185336_2520112_027/EMIT_L2A_RFL_001_20250720T185336_2520112_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T185336_2520112_027/EMIT_L2A_RFLUNCERT_001_20250720T185336_2520112_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T185336_2520112_027/EMIT_L2A_MASK_001_20250720T185336_2520112_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T185324_2520112_026/EMIT_L2A_RFL_001_20250720T185324_2520112_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T185324_2520112_026/EMIT_L2A_RFLUNCERT_001_20250720T185324_2520112_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250720T185324_2520112_026/EMIT_L2A_MASK_001_20250720T185324_2520112_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T211410_2520014_007/EMIT_L2A_RFL_001_20250719T211410_2520014_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T211410_2520014_007/EMIT_L2A_RFLUNCERT_001_20250719T211410_2520014_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T211410_2520014_007/EMIT_L2A_MASK_001_20250719T211410_2520014_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194148_2520013_012/EMIT_L2A_RFL_001_20250719T194148_2520013_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194148_2520013_012/EMIT_L2A_RFLUNCERT_001_20250719T194148_2520013_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194148_2520013_012/EMIT_L2A_MASK_001_20250719T194148_2520013_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194136_2520013_011/EMIT_L2A_RFL_001_20250719T194136_2520013_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194136_2520013_011/EMIT_L2A_RFLUNCERT_001_20250719T194136_2520013_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194136_2520013_011/EMIT_L2A_MASK_001_20250719T194136_2520013_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194125_2520013_010/EMIT_L2A_RFL_001_20250719T194125_2520013_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194125_2520013_010/EMIT_L2A_RFLUNCERT_001_20250719T194125_2520013_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194125_2520013_010/EMIT_L2A_MASK_001_20250719T194125_2520013_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194113_2520013_009/EMIT_L2A_RFL_001_20250719T194113_2520013_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194113_2520013_009/EMIT_L2A_RFLUNCERT_001_20250719T194113_2520013_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194113_2520013_009/EMIT_L2A_MASK_001_20250719T194113_2520013_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194101_2520013_008/EMIT_L2A_RFL_001_20250719T194101_2520013_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194101_2520013_008/EMIT_L2A_RFLUNCERT_001_20250719T194101_2520013_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194101_2520013_008/EMIT_L2A_MASK_001_20250719T194101_2520013_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194049_2520013_007/EMIT_L2A_RFL_001_20250719T194049_2520013_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194049_2520013_007/EMIT_L2A_RFLUNCERT_001_20250719T194049_2520013_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250719T194049_2520013_007/EMIT_L2A_MASK_001_20250719T194049_2520013_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202902_2519913_012/EMIT_L2A_RFL_001_20250718T202902_2519913_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202902_2519913_012/EMIT_L2A_RFLUNCERT_001_20250718T202902_2519913_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202902_2519913_012/EMIT_L2A_MASK_001_20250718T202902_2519913_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202850_2519913_011/EMIT_L2A_RFL_001_20250718T202850_2519913_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202850_2519913_011/EMIT_L2A_RFLUNCERT_001_20250718T202850_2519913_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202850_2519913_011/EMIT_L2A_MASK_001_20250718T202850_2519913_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202838_2519913_010/EMIT_L2A_RFL_001_20250718T202838_2519913_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202838_2519913_010/EMIT_L2A_RFLUNCERT_001_20250718T202838_2519913_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202838_2519913_010/EMIT_L2A_MASK_001_20250718T202838_2519913_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202826_2519913_009/EMIT_L2A_RFL_001_20250718T202826_2519913_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202826_2519913_009/EMIT_L2A_RFLUNCERT_001_20250718T202826_2519913_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250718T202826_2519913_009/EMIT_L2A_MASK_001_20250718T202826_2519913_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220653_2519714_017/EMIT_L2A_RFL_001_20250716T220653_2519714_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220653_2519714_017/EMIT_L2A_RFLUNCERT_001_20250716T220653_2519714_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220653_2519714_017/EMIT_L2A_MASK_001_20250716T220653_2519714_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220629_2519714_015/EMIT_L2A_RFL_001_20250716T220629_2519714_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220629_2519714_015/EMIT_L2A_RFLUNCERT_001_20250716T220629_2519714_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220629_2519714_015/EMIT_L2A_MASK_001_20250716T220629_2519714_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220413_2519714_014/EMIT_L2A_RFL_001_20250716T220413_2519714_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220413_2519714_014/EMIT_L2A_RFLUNCERT_001_20250716T220413_2519714_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220413_2519714_014/EMIT_L2A_MASK_001_20250716T220413_2519714_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220401_2519714_013/EMIT_L2A_RFL_001_20250716T220401_2519714_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220401_2519714_013/EMIT_L2A_RFLUNCERT_001_20250716T220401_2519714_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T220401_2519714_013/EMIT_L2A_MASK_001_20250716T220401_2519714_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T203015_2519713_028/EMIT_L2A_RFL_001_20250716T203015_2519713_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T203015_2519713_028/EMIT_L2A_RFLUNCERT_001_20250716T203015_2519713_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T203015_2519713_028/EMIT_L2A_MASK_001_20250716T203015_2519713_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T203003_2519713_027/EMIT_L2A_RFL_001_20250716T203003_2519713_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T203003_2519713_027/EMIT_L2A_RFLUNCERT_001_20250716T203003_2519713_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T203003_2519713_027/EMIT_L2A_MASK_001_20250716T203003_2519713_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T202951_2519713_026/EMIT_L2A_RFL_001_20250716T202951_2519713_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T202951_2519713_026/EMIT_L2A_RFLUNCERT_001_20250716T202951_2519713_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250716T202951_2519713_026/EMIT_L2A_MASK_001_20250716T202951_2519713_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250715T211751_2519614_008/EMIT_L2A_RFL_001_20250715T211751_2519614_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250715T211751_2519614_008/EMIT_L2A_RFLUNCERT_001_20250715T211751_2519614_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250715T211751_2519614_008/EMIT_L2A_MASK_001_20250715T211751_2519614_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250715T211739_2519614_007/EMIT_L2A_RFL_001_20250715T211739_2519614_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250715T211739_2519614_007/EMIT_L2A_RFLUNCERT_001_20250715T211739_2519614_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250715T211739_2519614_007/EMIT_L2A_MASK_001_20250715T211739_2519614_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125446_2518208_015/EMIT_L2A_RFL_001_20250701T125446_2518208_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125446_2518208_015/EMIT_L2A_RFLUNCERT_001_20250701T125446_2518208_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125446_2518208_015/EMIT_L2A_MASK_001_20250701T125446_2518208_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125434_2518208_014/EMIT_L2A_RFL_001_20250701T125434_2518208_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125434_2518208_014/EMIT_L2A_RFLUNCERT_001_20250701T125434_2518208_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125434_2518208_014/EMIT_L2A_MASK_001_20250701T125434_2518208_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125422_2518208_013/EMIT_L2A_RFL_001_20250701T125422_2518208_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125422_2518208_013/EMIT_L2A_RFLUNCERT_001_20250701T125422_2518208_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125422_2518208_013/EMIT_L2A_MASK_001_20250701T125422_2518208_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125410_2518208_012/EMIT_L2A_RFL_001_20250701T125410_2518208_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125410_2518208_012/EMIT_L2A_RFLUNCERT_001_20250701T125410_2518208_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125410_2518208_012/EMIT_L2A_MASK_001_20250701T125410_2518208_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125359_2518208_011/EMIT_L2A_RFL_001_20250701T125359_2518208_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125359_2518208_011/EMIT_L2A_RFLUNCERT_001_20250701T125359_2518208_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125359_2518208_011/EMIT_L2A_MASK_001_20250701T125359_2518208_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125347_2518208_010/EMIT_L2A_RFL_001_20250701T125347_2518208_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125347_2518208_010/EMIT_L2A_RFLUNCERT_001_20250701T125347_2518208_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125347_2518208_010/EMIT_L2A_MASK_001_20250701T125347_2518208_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125335_2518208_009/EMIT_L2A_RFL_001_20250701T125335_2518208_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125335_2518208_009/EMIT_L2A_RFLUNCERT_001_20250701T125335_2518208_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125335_2518208_009/EMIT_L2A_MASK_001_20250701T125335_2518208_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125323_2518208_008/EMIT_L2A_RFL_001_20250701T125323_2518208_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125323_2518208_008/EMIT_L2A_RFLUNCERT_001_20250701T125323_2518208_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125323_2518208_008/EMIT_L2A_MASK_001_20250701T125323_2518208_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125311_2518208_007/EMIT_L2A_RFL_001_20250701T125311_2518208_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125311_2518208_007/EMIT_L2A_RFLUNCERT_001_20250701T125311_2518208_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125311_2518208_007/EMIT_L2A_MASK_001_20250701T125311_2518208_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125259_2518208_006/EMIT_L2A_RFL_001_20250701T125259_2518208_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125259_2518208_006/EMIT_L2A_RFLUNCERT_001_20250701T125259_2518208_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125259_2518208_006/EMIT_L2A_MASK_001_20250701T125259_2518208_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125248_2518208_005/EMIT_L2A_RFL_001_20250701T125248_2518208_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125248_2518208_005/EMIT_L2A_RFLUNCERT_001_20250701T125248_2518208_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125248_2518208_005/EMIT_L2A_MASK_001_20250701T125248_2518208_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125236_2518208_004/EMIT_L2A_RFL_001_20250701T125236_2518208_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125236_2518208_004/EMIT_L2A_RFLUNCERT_001_20250701T125236_2518208_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125236_2518208_004/EMIT_L2A_MASK_001_20250701T125236_2518208_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125224_2518208_003/EMIT_L2A_RFL_001_20250701T125224_2518208_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125224_2518208_003/EMIT_L2A_RFLUNCERT_001_20250701T125224_2518208_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125224_2518208_003/EMIT_L2A_MASK_001_20250701T125224_2518208_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125212_2518208_002/EMIT_L2A_RFL_001_20250701T125212_2518208_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125212_2518208_002/EMIT_L2A_RFLUNCERT_001_20250701T125212_2518208_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250701T125212_2518208_002/EMIT_L2A_MASK_001_20250701T125212_2518208_002.nc
EDSCEOF

