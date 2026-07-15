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
    echo "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250403T080210_2509305_007/EMIT_L2A_RFL_001_20250403T080210_2509305_007.nc"
    echo
    exit 1
}

prompt_credentials
  detect_app_approval() {
    approved=`curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250403T080210_2509305_007/EMIT_L2A_RFL_001_20250403T080210_2509305_007.nc -w '\n%{http_code}' | tail  -1`
    if [ "$approved" -ne "200" ] && [ "$approved" -ne "301" ] && [ "$approved" -ne "302" ]; then
        # User didn't approve the app. Direct users to approve the app in URS
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below "
    fi
}

setup_auth_curl() {
    # Firstly, check if it require URS authentication
    status=$(curl -s -z "$(date)" -w '\n%{http_code}' https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250403T080210_2509305_007/EMIT_L2A_RFL_001_20250403T080210_2509305_007.nc | tail -1)
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
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250403T080210_2509305_007/EMIT_L2A_RFL_001_20250403T080210_2509305_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250403T080210_2509305_007/EMIT_L2A_RFLUNCERT_001_20250403T080210_2509305_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250403T080210_2509305_007/EMIT_L2A_MASK_001_20250403T080210_2509305_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071401_2509405_014/EMIT_L2A_RFL_001_20250404T071401_2509405_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071401_2509405_014/EMIT_L2A_RFLUNCERT_001_20250404T071401_2509405_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071401_2509405_014/EMIT_L2A_MASK_001_20250404T071401_2509405_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071413_2509405_015/EMIT_L2A_RFL_001_20250404T071413_2509405_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071413_2509405_015/EMIT_L2A_RFLUNCERT_001_20250404T071413_2509405_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071413_2509405_015/EMIT_L2A_MASK_001_20250404T071413_2509405_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071425_2509405_016/EMIT_L2A_RFL_001_20250404T071425_2509405_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071425_2509405_016/EMIT_L2A_RFLUNCERT_001_20250404T071425_2509405_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071425_2509405_016/EMIT_L2A_MASK_001_20250404T071425_2509405_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071437_2509405_017/EMIT_L2A_RFL_001_20250404T071437_2509405_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071437_2509405_017/EMIT_L2A_RFLUNCERT_001_20250404T071437_2509405_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250404T071437_2509405_017/EMIT_L2A_MASK_001_20250404T071437_2509405_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250419T095414_2510906_023/EMIT_L2A_RFL_001_20250419T095414_2510906_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250419T095414_2510906_023/EMIT_L2A_RFLUNCERT_001_20250419T095414_2510906_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250419T095414_2510906_023/EMIT_L2A_MASK_001_20250419T095414_2510906_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250422T091047_2511206_016/EMIT_L2A_RFL_001_20250422T091047_2511206_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250422T091047_2511206_016/EMIT_L2A_RFLUNCERT_001_20250422T091047_2511206_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250422T091047_2511206_016/EMIT_L2A_MASK_001_20250422T091047_2511206_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250426T073452_2511605_010/EMIT_L2A_RFL_001_20250426T073452_2511605_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250426T073452_2511605_010/EMIT_L2A_RFLUNCERT_001_20250426T073452_2511605_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250426T073452_2511605_010/EMIT_L2A_MASK_001_20250426T073452_2511605_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250426T073504_2511605_011/EMIT_L2A_RFL_001_20250426T073504_2511605_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250426T073504_2511605_011/EMIT_L2A_RFLUNCERT_001_20250426T073504_2511605_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250426T073504_2511605_011/EMIT_L2A_MASK_001_20250426T073504_2511605_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250426T073516_2511605_012/EMIT_L2A_RFL_001_20250426T073516_2511605_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250426T073516_2511605_012/EMIT_L2A_RFLUNCERT_001_20250426T073516_2511605_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250426T073516_2511605_012/EMIT_L2A_MASK_001_20250426T073516_2511605_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060042_2512004_002/EMIT_L2A_RFL_001_20250430T060042_2512004_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060042_2512004_002/EMIT_L2A_RFLUNCERT_001_20250430T060042_2512004_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060042_2512004_002/EMIT_L2A_MASK_001_20250430T060042_2512004_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060054_2512004_003/EMIT_L2A_RFL_001_20250430T060054_2512004_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060054_2512004_003/EMIT_L2A_RFLUNCERT_001_20250430T060054_2512004_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060054_2512004_003/EMIT_L2A_MASK_001_20250430T060054_2512004_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060106_2512004_004/EMIT_L2A_RFL_001_20250430T060106_2512004_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060106_2512004_004/EMIT_L2A_RFLUNCERT_001_20250430T060106_2512004_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060106_2512004_004/EMIT_L2A_MASK_001_20250430T060106_2512004_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060118_2512004_005/EMIT_L2A_RFL_001_20250430T060118_2512004_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060118_2512004_005/EMIT_L2A_RFLUNCERT_001_20250430T060118_2512004_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20250430T060118_2512004_005/EMIT_L2A_MASK_001_20250430T060118_2512004_005.nc
EDSCEOF
