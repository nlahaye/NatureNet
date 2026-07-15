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
    echo "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142812_2424209_010/EMIT_L2A_RFL_001_20240829T142812_2424209_010.nc"
    echo
    exit 1
}

prompt_credentials
  detect_app_approval() {
    approved=`curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142812_2424209_010/EMIT_L2A_RFL_001_20240829T142812_2424209_010.nc -w '\n%{http_code}' | tail  -1`
    if [ "$approved" -ne "200" ] && [ "$approved" -ne "301" ] && [ "$approved" -ne "302" ]; then
        # User didn't approve the app. Direct users to approve the app in URS
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below "
    fi
}

setup_auth_curl() {
    # Firstly, check if it require URS authentication
    status=$(curl -s -z "$(date)" -w '\n%{http_code}' https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142812_2424209_010/EMIT_L2A_RFL_001_20240829T142812_2424209_010.nc | tail -1)
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
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142812_2424209_010/EMIT_L2A_RFL_001_20240829T142812_2424209_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142812_2424209_010/EMIT_L2A_RFLUNCERT_001_20240829T142812_2424209_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142812_2424209_010/EMIT_L2A_MASK_001_20240829T142812_2424209_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142800_2424209_009/EMIT_L2A_RFL_001_20240829T142800_2424209_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142800_2424209_009/EMIT_L2A_RFLUNCERT_001_20240829T142800_2424209_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142800_2424209_009/EMIT_L2A_MASK_001_20240829T142800_2424209_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142748_2424209_008/EMIT_L2A_RFL_001_20240829T142748_2424209_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142748_2424209_008/EMIT_L2A_RFLUNCERT_001_20240829T142748_2424209_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142748_2424209_008/EMIT_L2A_MASK_001_20240829T142748_2424209_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142736_2424209_007/EMIT_L2A_RFL_001_20240829T142736_2424209_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142736_2424209_007/EMIT_L2A_RFLUNCERT_001_20240829T142736_2424209_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142736_2424209_007/EMIT_L2A_MASK_001_20240829T142736_2424209_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142724_2424209_006/EMIT_L2A_RFL_001_20240829T142724_2424209_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142724_2424209_006/EMIT_L2A_RFLUNCERT_001_20240829T142724_2424209_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142724_2424209_006/EMIT_L2A_MASK_001_20240829T142724_2424209_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142713_2424209_005/EMIT_L2A_RFL_001_20240829T142713_2424209_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142713_2424209_005/EMIT_L2A_RFLUNCERT_001_20240829T142713_2424209_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142713_2424209_005/EMIT_L2A_MASK_001_20240829T142713_2424209_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142701_2424209_004/EMIT_L2A_RFL_001_20240829T142701_2424209_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142701_2424209_004/EMIT_L2A_RFLUNCERT_001_20240829T142701_2424209_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142701_2424209_004/EMIT_L2A_MASK_001_20240829T142701_2424209_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142649_2424209_003/EMIT_L2A_RFL_001_20240829T142649_2424209_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142649_2424209_003/EMIT_L2A_RFLUNCERT_001_20240829T142649_2424209_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142649_2424209_003/EMIT_L2A_MASK_001_20240829T142649_2424209_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142637_2424209_002/EMIT_L2A_RFL_001_20240829T142637_2424209_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142637_2424209_002/EMIT_L2A_RFLUNCERT_001_20240829T142637_2424209_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240829T142637_2424209_002/EMIT_L2A_MASK_001_20240829T142637_2424209_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240828T151452_2424110_009/EMIT_L2A_RFL_001_20240828T151452_2424110_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240828T151452_2424110_009/EMIT_L2A_RFLUNCERT_001_20240828T151452_2424110_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240828T151452_2424110_009/EMIT_L2A_MASK_001_20240828T151452_2424110_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151553_2423910_027/EMIT_L2A_RFL_001_20240826T151553_2423910_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151553_2423910_027/EMIT_L2A_RFLUNCERT_001_20240826T151553_2423910_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151553_2423910_027/EMIT_L2A_MASK_001_20240826T151553_2423910_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151541_2423910_026/EMIT_L2A_RFL_001_20240826T151541_2423910_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151541_2423910_026/EMIT_L2A_RFLUNCERT_001_20240826T151541_2423910_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151541_2423910_026/EMIT_L2A_MASK_001_20240826T151541_2423910_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151529_2423910_025/EMIT_L2A_RFL_001_20240826T151529_2423910_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151529_2423910_025/EMIT_L2A_RFLUNCERT_001_20240826T151529_2423910_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151529_2423910_025/EMIT_L2A_MASK_001_20240826T151529_2423910_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151517_2423910_024/EMIT_L2A_RFL_001_20240826T151517_2423910_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151517_2423910_024/EMIT_L2A_RFLUNCERT_001_20240826T151517_2423910_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151517_2423910_024/EMIT_L2A_MASK_001_20240826T151517_2423910_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151446_2423910_023/EMIT_L2A_RFL_001_20240826T151446_2423910_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151446_2423910_023/EMIT_L2A_RFLUNCERT_001_20240826T151446_2423910_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151446_2423910_023/EMIT_L2A_MASK_001_20240826T151446_2423910_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151434_2423910_022/EMIT_L2A_RFL_001_20240826T151434_2423910_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151434_2423910_022/EMIT_L2A_RFLUNCERT_001_20240826T151434_2423910_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151434_2423910_022/EMIT_L2A_MASK_001_20240826T151434_2423910_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151422_2423910_021/EMIT_L2A_RFL_001_20240826T151422_2423910_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151422_2423910_021/EMIT_L2A_RFLUNCERT_001_20240826T151422_2423910_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151422_2423910_021/EMIT_L2A_MASK_001_20240826T151422_2423910_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151410_2423910_020/EMIT_L2A_RFL_001_20240826T151410_2423910_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151410_2423910_020/EMIT_L2A_RFLUNCERT_001_20240826T151410_2423910_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151410_2423910_020/EMIT_L2A_MASK_001_20240826T151410_2423910_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151358_2423910_019/EMIT_L2A_RFL_001_20240826T151358_2423910_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151358_2423910_019/EMIT_L2A_RFLUNCERT_001_20240826T151358_2423910_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151358_2423910_019/EMIT_L2A_MASK_001_20240826T151358_2423910_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151347_2423910_018/EMIT_L2A_RFL_001_20240826T151347_2423910_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151347_2423910_018/EMIT_L2A_RFLUNCERT_001_20240826T151347_2423910_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151347_2423910_018/EMIT_L2A_MASK_001_20240826T151347_2423910_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151335_2423910_017/EMIT_L2A_RFL_001_20240826T151335_2423910_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151335_2423910_017/EMIT_L2A_RFLUNCERT_001_20240826T151335_2423910_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151335_2423910_017/EMIT_L2A_MASK_001_20240826T151335_2423910_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151323_2423910_016/EMIT_L2A_RFL_001_20240826T151323_2423910_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151323_2423910_016/EMIT_L2A_RFLUNCERT_001_20240826T151323_2423910_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151323_2423910_016/EMIT_L2A_MASK_001_20240826T151323_2423910_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151311_2423910_015/EMIT_L2A_RFL_001_20240826T151311_2423910_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151311_2423910_015/EMIT_L2A_RFLUNCERT_001_20240826T151311_2423910_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151311_2423910_015/EMIT_L2A_MASK_001_20240826T151311_2423910_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151259_2423910_014/EMIT_L2A_RFL_001_20240826T151259_2423910_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151259_2423910_014/EMIT_L2A_RFLUNCERT_001_20240826T151259_2423910_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151259_2423910_014/EMIT_L2A_MASK_001_20240826T151259_2423910_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151247_2423910_013/EMIT_L2A_RFL_001_20240826T151247_2423910_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151247_2423910_013/EMIT_L2A_RFLUNCERT_001_20240826T151247_2423910_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151247_2423910_013/EMIT_L2A_MASK_001_20240826T151247_2423910_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151236_2423910_012/EMIT_L2A_RFL_001_20240826T151236_2423910_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151236_2423910_012/EMIT_L2A_RFLUNCERT_001_20240826T151236_2423910_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151236_2423910_012/EMIT_L2A_MASK_001_20240826T151236_2423910_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151224_2423910_011/EMIT_L2A_RFL_001_20240826T151224_2423910_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151224_2423910_011/EMIT_L2A_RFLUNCERT_001_20240826T151224_2423910_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151224_2423910_011/EMIT_L2A_MASK_001_20240826T151224_2423910_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151212_2423910_010/EMIT_L2A_RFL_001_20240826T151212_2423910_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151212_2423910_010/EMIT_L2A_RFLUNCERT_001_20240826T151212_2423910_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240826T151212_2423910_010/EMIT_L2A_MASK_001_20240826T151212_2423910_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160342_2423810_028/EMIT_L2A_RFL_001_20240825T160342_2423810_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160342_2423810_028/EMIT_L2A_RFLUNCERT_001_20240825T160342_2423810_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160342_2423810_028/EMIT_L2A_MASK_001_20240825T160342_2423810_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160330_2423810_027/EMIT_L2A_RFL_001_20240825T160330_2423810_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160330_2423810_027/EMIT_L2A_RFLUNCERT_001_20240825T160330_2423810_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160330_2423810_027/EMIT_L2A_MASK_001_20240825T160330_2423810_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160318_2423810_026/EMIT_L2A_RFL_001_20240825T160318_2423810_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160318_2423810_026/EMIT_L2A_RFLUNCERT_001_20240825T160318_2423810_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160318_2423810_026/EMIT_L2A_MASK_001_20240825T160318_2423810_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160306_2423810_025/EMIT_L2A_RFL_001_20240825T160306_2423810_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160306_2423810_025/EMIT_L2A_RFLUNCERT_001_20240825T160306_2423810_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160306_2423810_025/EMIT_L2A_MASK_001_20240825T160306_2423810_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160255_2423810_024/EMIT_L2A_RFL_001_20240825T160255_2423810_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160255_2423810_024/EMIT_L2A_RFLUNCERT_001_20240825T160255_2423810_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160255_2423810_024/EMIT_L2A_MASK_001_20240825T160255_2423810_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160243_2423810_023/EMIT_L2A_RFL_001_20240825T160243_2423810_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160243_2423810_023/EMIT_L2A_RFLUNCERT_001_20240825T160243_2423810_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160243_2423810_023/EMIT_L2A_MASK_001_20240825T160243_2423810_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160231_2423810_022/EMIT_L2A_RFL_001_20240825T160231_2423810_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160231_2423810_022/EMIT_L2A_RFLUNCERT_001_20240825T160231_2423810_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160231_2423810_022/EMIT_L2A_MASK_001_20240825T160231_2423810_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160219_2423810_021/EMIT_L2A_RFL_001_20240825T160219_2423810_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160219_2423810_021/EMIT_L2A_RFLUNCERT_001_20240825T160219_2423810_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160219_2423810_021/EMIT_L2A_MASK_001_20240825T160219_2423810_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160207_2423810_020/EMIT_L2A_RFL_001_20240825T160207_2423810_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160207_2423810_020/EMIT_L2A_RFLUNCERT_001_20240825T160207_2423810_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240825T160207_2423810_020/EMIT_L2A_MASK_001_20240825T160207_2423810_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240821T174019_2423412_041/EMIT_L2A_RFL_001_20240821T174019_2423412_041.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240821T174019_2423412_041/EMIT_L2A_RFLUNCERT_001_20240821T174019_2423412_041.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240821T174019_2423412_041/EMIT_L2A_MASK_001_20240821T174019_2423412_041.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240821T174007_2423412_040/EMIT_L2A_RFL_001_20240821T174007_2423412_040.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240821T174007_2423412_040/EMIT_L2A_RFLUNCERT_001_20240821T174007_2423412_040.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240821T174007_2423412_040/EMIT_L2A_MASK_001_20240821T174007_2423412_040.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240817T191801_2423013_049/EMIT_L2A_RFL_001_20240817T191801_2423013_049.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240817T191801_2423013_049/EMIT_L2A_RFLUNCERT_001_20240817T191801_2423013_049.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240817T191801_2423013_049/EMIT_L2A_MASK_001_20240817T191801_2423013_049.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240817T191749_2423013_048/EMIT_L2A_RFL_001_20240817T191749_2423013_048.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240817T191749_2423013_048/EMIT_L2A_RFLUNCERT_001_20240817T191749_2423013_048.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240817T191749_2423013_048/EMIT_L2A_MASK_001_20240817T191749_2423013_048.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191940_2422813_055/EMIT_L2A_RFL_001_20240815T191940_2422813_055.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191940_2422813_055/EMIT_L2A_RFLUNCERT_001_20240815T191940_2422813_055.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191940_2422813_055/EMIT_L2A_MASK_001_20240815T191940_2422813_055.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191928_2422813_054/EMIT_L2A_RFL_001_20240815T191928_2422813_054.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191928_2422813_054/EMIT_L2A_RFLUNCERT_001_20240815T191928_2422813_054.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191928_2422813_054/EMIT_L2A_MASK_001_20240815T191928_2422813_054.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191916_2422813_053/EMIT_L2A_RFL_001_20240815T191916_2422813_053.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191916_2422813_053/EMIT_L2A_RFLUNCERT_001_20240815T191916_2422813_053.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191916_2422813_053/EMIT_L2A_MASK_001_20240815T191916_2422813_053.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191905_2422813_052/EMIT_L2A_RFL_001_20240815T191905_2422813_052.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191905_2422813_052/EMIT_L2A_RFLUNCERT_001_20240815T191905_2422813_052.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191905_2422813_052/EMIT_L2A_MASK_001_20240815T191905_2422813_052.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191853_2422813_051/EMIT_L2A_RFL_001_20240815T191853_2422813_051.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191853_2422813_051/EMIT_L2A_RFLUNCERT_001_20240815T191853_2422813_051.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191853_2422813_051/EMIT_L2A_MASK_001_20240815T191853_2422813_051.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191841_2422813_050/EMIT_L2A_RFL_001_20240815T191841_2422813_050.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191841_2422813_050/EMIT_L2A_RFLUNCERT_001_20240815T191841_2422813_050.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191841_2422813_050/EMIT_L2A_MASK_001_20240815T191841_2422813_050.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191829_2422813_049/EMIT_L2A_RFL_001_20240815T191829_2422813_049.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191829_2422813_049/EMIT_L2A_RFLUNCERT_001_20240815T191829_2422813_049.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240815T191829_2422813_049/EMIT_L2A_MASK_001_20240815T191829_2422813_049.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240805T141946_2421810_002/EMIT_L2A_RFL_001_20240805T141946_2421810_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240805T141946_2421810_002/EMIT_L2A_RFLUNCERT_001_20240805T141946_2421810_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240805T141946_2421810_002/EMIT_L2A_MASK_001_20240805T141946_2421810_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240805T141934_2421810_001/EMIT_L2A_RFL_001_20240805T141934_2421810_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240805T141934_2421810_001/EMIT_L2A_RFLUNCERT_001_20240805T141934_2421810_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240805T141934_2421810_001/EMIT_L2A_MASK_001_20240805T141934_2421810_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150747_2421710_013/EMIT_L2A_RFL_001_20240804T150747_2421710_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150747_2421710_013/EMIT_L2A_RFLUNCERT_001_20240804T150747_2421710_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150747_2421710_013/EMIT_L2A_MASK_001_20240804T150747_2421710_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150735_2421710_012/EMIT_L2A_RFL_001_20240804T150735_2421710_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150735_2421710_012/EMIT_L2A_RFLUNCERT_001_20240804T150735_2421710_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150735_2421710_012/EMIT_L2A_MASK_001_20240804T150735_2421710_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150724_2421710_011/EMIT_L2A_RFL_001_20240804T150724_2421710_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150724_2421710_011/EMIT_L2A_RFLUNCERT_001_20240804T150724_2421710_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150724_2421710_011/EMIT_L2A_MASK_001_20240804T150724_2421710_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150712_2421710_010/EMIT_L2A_RFL_001_20240804T150712_2421710_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150712_2421710_010/EMIT_L2A_RFLUNCERT_001_20240804T150712_2421710_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150712_2421710_010/EMIT_L2A_MASK_001_20240804T150712_2421710_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150700_2421710_009/EMIT_L2A_RFL_001_20240804T150700_2421710_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150700_2421710_009/EMIT_L2A_RFLUNCERT_001_20240804T150700_2421710_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150700_2421710_009/EMIT_L2A_MASK_001_20240804T150700_2421710_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150648_2421710_008/EMIT_L2A_RFL_001_20240804T150648_2421710_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150648_2421710_008/EMIT_L2A_RFLUNCERT_001_20240804T150648_2421710_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150648_2421710_008/EMIT_L2A_MASK_001_20240804T150648_2421710_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150636_2421710_007/EMIT_L2A_RFL_001_20240804T150636_2421710_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150636_2421710_007/EMIT_L2A_RFLUNCERT_001_20240804T150636_2421710_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150636_2421710_007/EMIT_L2A_MASK_001_20240804T150636_2421710_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150624_2421710_006/EMIT_L2A_RFL_001_20240804T150624_2421710_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150624_2421710_006/EMIT_L2A_RFLUNCERT_001_20240804T150624_2421710_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240804T150624_2421710_006/EMIT_L2A_MASK_001_20240804T150624_2421710_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155540_2421611_015/EMIT_L2A_RFL_001_20240803T155540_2421611_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155540_2421611_015/EMIT_L2A_RFLUNCERT_001_20240803T155540_2421611_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155540_2421611_015/EMIT_L2A_MASK_001_20240803T155540_2421611_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155528_2421611_014/EMIT_L2A_RFL_001_20240803T155528_2421611_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155528_2421611_014/EMIT_L2A_MASK_001_20240803T155528_2421611_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155528_2421611_014/EMIT_L2A_RFLUNCERT_001_20240803T155528_2421611_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155516_2421611_013/EMIT_L2A_RFL_001_20240803T155516_2421611_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155516_2421611_013/EMIT_L2A_MASK_001_20240803T155516_2421611_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155516_2421611_013/EMIT_L2A_RFLUNCERT_001_20240803T155516_2421611_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155504_2421611_012/EMIT_L2A_RFL_001_20240803T155504_2421611_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155504_2421611_012/EMIT_L2A_RFLUNCERT_001_20240803T155504_2421611_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240803T155504_2421611_012/EMIT_L2A_MASK_001_20240803T155504_2421611_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240802T150616_2421510_006/EMIT_L2A_RFL_001_20240802T150616_2421510_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240802T150616_2421510_006/EMIT_L2A_RFLUNCERT_001_20240802T150616_2421510_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240802T150616_2421510_006/EMIT_L2A_MASK_001_20240802T150616_2421510_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240802T150604_2421510_005/EMIT_L2A_RFL_001_20240802T150604_2421510_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240802T150604_2421510_005/EMIT_L2A_MASK_001_20240802T150604_2421510_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240802T150604_2421510_005/EMIT_L2A_RFLUNCERT_001_20240802T150604_2421510_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155503_2421411_011/EMIT_L2A_RFL_001_20240801T155503_2421411_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155503_2421411_011/EMIT_L2A_MASK_001_20240801T155503_2421411_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155503_2421411_011/EMIT_L2A_RFLUNCERT_001_20240801T155503_2421411_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155451_2421411_010/EMIT_L2A_RFL_001_20240801T155451_2421411_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155451_2421411_010/EMIT_L2A_MASK_001_20240801T155451_2421411_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155451_2421411_010/EMIT_L2A_RFLUNCERT_001_20240801T155451_2421411_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155439_2421411_009/EMIT_L2A_RFL_001_20240801T155439_2421411_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155439_2421411_009/EMIT_L2A_MASK_001_20240801T155439_2421411_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155439_2421411_009/EMIT_L2A_RFLUNCERT_001_20240801T155439_2421411_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155341_2421411_008/EMIT_L2A_RFL_001_20240801T155341_2421411_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155341_2421411_008/EMIT_L2A_MASK_001_20240801T155341_2421411_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155341_2421411_008/EMIT_L2A_RFLUNCERT_001_20240801T155341_2421411_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155329_2421411_007/EMIT_L2A_RFL_001_20240801T155329_2421411_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155329_2421411_007/EMIT_L2A_MASK_001_20240801T155329_2421411_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240801T155329_2421411_007/EMIT_L2A_RFLUNCERT_001_20240801T155329_2421411_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164258_2421311_012/EMIT_L2A_RFL_001_20240731T164258_2421311_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164258_2421311_012/EMIT_L2A_RFLUNCERT_001_20240731T164258_2421311_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164258_2421311_012/EMIT_L2A_MASK_001_20240731T164258_2421311_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164247_2421311_011/EMIT_L2A_RFL_001_20240731T164247_2421311_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164247_2421311_011/EMIT_L2A_MASK_001_20240731T164247_2421311_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164247_2421311_011/EMIT_L2A_RFLUNCERT_001_20240731T164247_2421311_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164235_2421311_010/EMIT_L2A_RFL_001_20240731T164235_2421311_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164235_2421311_010/EMIT_L2A_MASK_001_20240731T164235_2421311_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164235_2421311_010/EMIT_L2A_RFLUNCERT_001_20240731T164235_2421311_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164223_2421311_009/EMIT_L2A_RFL_001_20240731T164223_2421311_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164223_2421311_009/EMIT_L2A_MASK_001_20240731T164223_2421311_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164223_2421311_009/EMIT_L2A_RFLUNCERT_001_20240731T164223_2421311_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164211_2421311_008/EMIT_L2A_RFL_001_20240731T164211_2421311_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164211_2421311_008/EMIT_L2A_MASK_001_20240731T164211_2421311_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164211_2421311_008/EMIT_L2A_RFLUNCERT_001_20240731T164211_2421311_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164159_2421311_007/EMIT_L2A_RFL_001_20240731T164159_2421311_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164159_2421311_007/EMIT_L2A_MASK_001_20240731T164159_2421311_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240731T164159_2421311_007/EMIT_L2A_RFLUNCERT_001_20240731T164159_2421311_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240730T173120_2421212_009/EMIT_L2A_RFL_001_20240730T173120_2421212_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240730T173120_2421212_009/EMIT_L2A_MASK_001_20240730T173120_2421212_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240730T173120_2421212_009/EMIT_L2A_RFLUNCERT_001_20240730T173120_2421212_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240730T173108_2421212_008/EMIT_L2A_RFL_001_20240730T173108_2421212_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240730T173108_2421212_008/EMIT_L2A_MASK_001_20240730T173108_2421212_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240730T173108_2421212_008/EMIT_L2A_RFLUNCERT_001_20240730T173108_2421212_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240729T164156_2421111_007/EMIT_L2A_RFL_001_20240729T164156_2421111_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240729T164156_2421111_007/EMIT_L2A_RFLUNCERT_001_20240729T164156_2421111_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240729T164156_2421111_007/EMIT_L2A_MASK_001_20240729T164156_2421111_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173236_2421012_011/EMIT_L2A_RFL_001_20240728T173236_2421012_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173236_2421012_011/EMIT_L2A_MASK_001_20240728T173236_2421012_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173236_2421012_011/EMIT_L2A_RFLUNCERT_001_20240728T173236_2421012_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173224_2421012_010/EMIT_L2A_RFL_001_20240728T173224_2421012_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173224_2421012_010/EMIT_L2A_RFLUNCERT_001_20240728T173224_2421012_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173224_2421012_010/EMIT_L2A_MASK_001_20240728T173224_2421012_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173136_2421012_009/EMIT_L2A_RFL_001_20240728T173136_2421012_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173136_2421012_009/EMIT_L2A_MASK_001_20240728T173136_2421012_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173136_2421012_009/EMIT_L2A_RFLUNCERT_001_20240728T173136_2421012_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173124_2421012_008/EMIT_L2A_RFL_001_20240728T173124_2421012_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173124_2421012_008/EMIT_L2A_MASK_001_20240728T173124_2421012_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240728T173124_2421012_008/EMIT_L2A_RFLUNCERT_001_20240728T173124_2421012_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182119_2420912_008/EMIT_L2A_RFL_001_20240727T182119_2420912_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182119_2420912_008/EMIT_L2A_MASK_001_20240727T182119_2420912_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182119_2420912_008/EMIT_L2A_RFLUNCERT_001_20240727T182119_2420912_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182107_2420912_007/EMIT_L2A_RFL_001_20240727T182107_2420912_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182107_2420912_007/EMIT_L2A_MASK_001_20240727T182107_2420912_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182107_2420912_007/EMIT_L2A_RFLUNCERT_001_20240727T182107_2420912_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182055_2420912_006/EMIT_L2A_RFL_001_20240727T182055_2420912_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182055_2420912_006/EMIT_L2A_MASK_001_20240727T182055_2420912_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182055_2420912_006/EMIT_L2A_RFLUNCERT_001_20240727T182055_2420912_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182043_2420912_005/EMIT_L2A_RFL_001_20240727T182043_2420912_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182043_2420912_005/EMIT_L2A_MASK_001_20240727T182043_2420912_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182043_2420912_005/EMIT_L2A_RFLUNCERT_001_20240727T182043_2420912_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182031_2420912_004/EMIT_L2A_RFL_001_20240727T182031_2420912_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182031_2420912_004/EMIT_L2A_MASK_001_20240727T182031_2420912_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182031_2420912_004/EMIT_L2A_RFLUNCERT_001_20240727T182031_2420912_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182019_2420912_003/EMIT_L2A_RFL_001_20240727T182019_2420912_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182019_2420912_003/EMIT_L2A_MASK_001_20240727T182019_2420912_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240727T182019_2420912_003/EMIT_L2A_RFLUNCERT_001_20240727T182019_2420912_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240725T182043_2420712_008/EMIT_L2A_RFL_001_20240725T182043_2420712_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240725T182043_2420712_008/EMIT_L2A_RFLUNCERT_001_20240725T182043_2420712_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240725T182043_2420712_008/EMIT_L2A_MASK_001_20240725T182043_2420712_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240725T182031_2420712_007/EMIT_L2A_RFL_001_20240725T182031_2420712_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240725T182031_2420712_007/EMIT_L2A_RFLUNCERT_001_20240725T182031_2420712_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240725T182031_2420712_007/EMIT_L2A_MASK_001_20240725T182031_2420712_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240725T182019_2420712_006/EMIT_L2A_RFL_001_20240725T182019_2420712_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240725T182019_2420712_006/EMIT_L2A_RFLUNCERT_001_20240725T182019_2420712_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240725T182019_2420712_006/EMIT_L2A_MASK_001_20240725T182019_2420712_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195918_2420513_023/EMIT_L2A_RFL_001_20240723T195918_2420513_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195918_2420513_023/EMIT_L2A_RFLUNCERT_001_20240723T195918_2420513_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195918_2420513_023/EMIT_L2A_MASK_001_20240723T195918_2420513_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195907_2420513_022/EMIT_L2A_RFL_001_20240723T195907_2420513_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195907_2420513_022/EMIT_L2A_RFLUNCERT_001_20240723T195907_2420513_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195907_2420513_022/EMIT_L2A_MASK_001_20240723T195907_2420513_022.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195855_2420513_021/EMIT_L2A_RFL_001_20240723T195855_2420513_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195855_2420513_021/EMIT_L2A_RFLUNCERT_001_20240723T195855_2420513_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195855_2420513_021/EMIT_L2A_MASK_001_20240723T195855_2420513_021.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195843_2420513_020/EMIT_L2A_RFL_001_20240723T195843_2420513_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195843_2420513_020/EMIT_L2A_RFLUNCERT_001_20240723T195843_2420513_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195843_2420513_020/EMIT_L2A_MASK_001_20240723T195843_2420513_020.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195831_2420513_019/EMIT_L2A_RFL_001_20240723T195831_2420513_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195831_2420513_019/EMIT_L2A_RFLUNCERT_001_20240723T195831_2420513_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195831_2420513_019/EMIT_L2A_MASK_001_20240723T195831_2420513_019.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195819_2420513_018/EMIT_L2A_RFL_001_20240723T195819_2420513_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195819_2420513_018/EMIT_L2A_RFLUNCERT_001_20240723T195819_2420513_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240723T195819_2420513_018/EMIT_L2A_MASK_001_20240723T195819_2420513_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195929_2420313_018/EMIT_L2A_RFL_001_20240721T195929_2420313_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195929_2420313_018/EMIT_L2A_RFLUNCERT_001_20240721T195929_2420313_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195929_2420313_018/EMIT_L2A_MASK_001_20240721T195929_2420313_018.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195917_2420313_017/EMIT_L2A_RFL_001_20240721T195917_2420313_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195917_2420313_017/EMIT_L2A_MASK_001_20240721T195917_2420313_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195917_2420313_017/EMIT_L2A_RFLUNCERT_001_20240721T195917_2420313_017.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195905_2420313_016/EMIT_L2A_RFL_001_20240721T195905_2420313_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195905_2420313_016/EMIT_L2A_RFLUNCERT_001_20240721T195905_2420313_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195905_2420313_016/EMIT_L2A_MASK_001_20240721T195905_2420313_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195853_2420313_015/EMIT_L2A_RFL_001_20240721T195853_2420313_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195853_2420313_015/EMIT_L2A_MASK_001_20240721T195853_2420313_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195853_2420313_015/EMIT_L2A_RFLUNCERT_001_20240721T195853_2420313_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195842_2420313_014/EMIT_L2A_RFL_001_20240721T195842_2420313_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195842_2420313_014/EMIT_L2A_RFLUNCERT_001_20240721T195842_2420313_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195842_2420313_014/EMIT_L2A_MASK_001_20240721T195842_2420313_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195830_2420313_013/EMIT_L2A_RFL_001_20240721T195830_2420313_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195830_2420313_013/EMIT_L2A_RFLUNCERT_001_20240721T195830_2420313_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195830_2420313_013/EMIT_L2A_MASK_001_20240721T195830_2420313_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195806_2420313_011/EMIT_L2A_RFL_001_20240721T195806_2420313_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195806_2420313_011/EMIT_L2A_RFLUNCERT_001_20240721T195806_2420313_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195806_2420313_011/EMIT_L2A_MASK_001_20240721T195806_2420313_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195754_2420313_010/EMIT_L2A_RFL_001_20240721T195754_2420313_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195754_2420313_010/EMIT_L2A_RFLUNCERT_001_20240721T195754_2420313_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240721T195754_2420313_010/EMIT_L2A_MASK_001_20240721T195754_2420313_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204700_2420214_016/EMIT_L2A_RFL_001_20240720T204700_2420214_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204700_2420214_016/EMIT_L2A_RFLUNCERT_001_20240720T204700_2420214_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204700_2420214_016/EMIT_L2A_MASK_001_20240720T204700_2420214_016.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204649_2420214_015/EMIT_L2A_RFL_001_20240720T204649_2420214_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204649_2420214_015/EMIT_L2A_RFLUNCERT_001_20240720T204649_2420214_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204649_2420214_015/EMIT_L2A_MASK_001_20240720T204649_2420214_015.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204637_2420214_014/EMIT_L2A_RFL_001_20240720T204637_2420214_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204637_2420214_014/EMIT_L2A_RFLUNCERT_001_20240720T204637_2420214_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204637_2420214_014/EMIT_L2A_MASK_001_20240720T204637_2420214_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204625_2420214_013/EMIT_L2A_RFL_001_20240720T204625_2420214_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204625_2420214_013/EMIT_L2A_RFLUNCERT_001_20240720T204625_2420214_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204625_2420214_013/EMIT_L2A_MASK_001_20240720T204625_2420214_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204613_2420214_012/EMIT_L2A_RFL_001_20240720T204613_2420214_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204613_2420214_012/EMIT_L2A_RFLUNCERT_001_20240720T204613_2420214_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204613_2420214_012/EMIT_L2A_MASK_001_20240720T204613_2420214_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204601_2420214_011/EMIT_L2A_RFL_001_20240720T204601_2420214_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204601_2420214_011/EMIT_L2A_RFLUNCERT_001_20240720T204601_2420214_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204601_2420214_011/EMIT_L2A_MASK_001_20240720T204601_2420214_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204549_2420214_010/EMIT_L2A_RFL_001_20240720T204549_2420214_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204549_2420214_010/EMIT_L2A_MASK_001_20240720T204549_2420214_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240720T204549_2420214_010/EMIT_L2A_RFLUNCERT_001_20240720T204549_2420214_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240718T204646_2420014_005/EMIT_L2A_RFL_001_20240718T204646_2420014_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240718T204646_2420014_005/EMIT_L2A_RFLUNCERT_001_20240718T204646_2420014_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240718T204646_2420014_005/EMIT_L2A_MASK_001_20240718T204646_2420014_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240702T134716_2418409_007/EMIT_L2A_RFL_001_20240702T134716_2418409_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240702T134716_2418409_007/EMIT_L2A_RFLUNCERT_001_20240702T134716_2418409_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240702T134716_2418409_007/EMIT_L2A_MASK_001_20240702T134716_2418409_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143554_2418110_011/EMIT_L2A_RFL_001_20240629T143554_2418110_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143554_2418110_011/EMIT_L2A_MASK_001_20240629T143554_2418110_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143554_2418110_011/EMIT_L2A_RFLUNCERT_001_20240629T143554_2418110_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143542_2418110_010/EMIT_L2A_RFL_001_20240629T143542_2418110_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143542_2418110_010/EMIT_L2A_MASK_001_20240629T143542_2418110_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143542_2418110_010/EMIT_L2A_RFLUNCERT_001_20240629T143542_2418110_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143530_2418110_009/EMIT_L2A_RFL_001_20240629T143530_2418110_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143530_2418110_009/EMIT_L2A_MASK_001_20240629T143530_2418110_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143530_2418110_009/EMIT_L2A_RFLUNCERT_001_20240629T143530_2418110_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143518_2418110_008/EMIT_L2A_RFL_001_20240629T143518_2418110_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143518_2418110_008/EMIT_L2A_MASK_001_20240629T143518_2418110_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143518_2418110_008/EMIT_L2A_RFLUNCERT_001_20240629T143518_2418110_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143506_2418110_007/EMIT_L2A_RFL_001_20240629T143506_2418110_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143506_2418110_007/EMIT_L2A_MASK_001_20240629T143506_2418110_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143506_2418110_007/EMIT_L2A_RFLUNCERT_001_20240629T143506_2418110_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143454_2418110_006/EMIT_L2A_RFL_001_20240629T143454_2418110_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143454_2418110_006/EMIT_L2A_MASK_001_20240629T143454_2418110_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143454_2418110_006/EMIT_L2A_RFLUNCERT_001_20240629T143454_2418110_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143443_2418110_005/EMIT_L2A_RFL_001_20240629T143443_2418110_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143443_2418110_005/EMIT_L2A_MASK_001_20240629T143443_2418110_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143443_2418110_005/EMIT_L2A_RFLUNCERT_001_20240629T143443_2418110_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143431_2418110_004/EMIT_L2A_RFL_001_20240629T143431_2418110_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143431_2418110_004/EMIT_L2A_MASK_001_20240629T143431_2418110_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143431_2418110_004/EMIT_L2A_RFLUNCERT_001_20240629T143431_2418110_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143419_2418110_003/EMIT_L2A_RFL_001_20240629T143419_2418110_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143419_2418110_003/EMIT_L2A_MASK_001_20240629T143419_2418110_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143419_2418110_003/EMIT_L2A_RFLUNCERT_001_20240629T143419_2418110_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143407_2418110_002/EMIT_L2A_RFL_001_20240629T143407_2418110_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143407_2418110_002/EMIT_L2A_MASK_001_20240629T143407_2418110_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240629T143407_2418110_002/EMIT_L2A_RFLUNCERT_001_20240629T143407_2418110_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152255_2417810_036/EMIT_L2A_RFL_001_20240626T152255_2417810_036.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152255_2417810_036/EMIT_L2A_RFLUNCERT_001_20240626T152255_2417810_036.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152255_2417810_036/EMIT_L2A_MASK_001_20240626T152255_2417810_036.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152243_2417810_035/EMIT_L2A_RFL_001_20240626T152243_2417810_035.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152243_2417810_035/EMIT_L2A_RFLUNCERT_001_20240626T152243_2417810_035.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152243_2417810_035/EMIT_L2A_MASK_001_20240626T152243_2417810_035.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152231_2417810_034/EMIT_L2A_RFL_001_20240626T152231_2417810_034.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152231_2417810_034/EMIT_L2A_RFLUNCERT_001_20240626T152231_2417810_034.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152231_2417810_034/EMIT_L2A_MASK_001_20240626T152231_2417810_034.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152219_2417810_033/EMIT_L2A_RFL_001_20240626T152219_2417810_033.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152219_2417810_033/EMIT_L2A_RFLUNCERT_001_20240626T152219_2417810_033.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152219_2417810_033/EMIT_L2A_MASK_001_20240626T152219_2417810_033.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152207_2417810_032/EMIT_L2A_RFL_001_20240626T152207_2417810_032.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152207_2417810_032/EMIT_L2A_RFLUNCERT_001_20240626T152207_2417810_032.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152207_2417810_032/EMIT_L2A_MASK_001_20240626T152207_2417810_032.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152155_2417810_031/EMIT_L2A_RFL_001_20240626T152155_2417810_031.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152155_2417810_031/EMIT_L2A_RFLUNCERT_001_20240626T152155_2417810_031.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152155_2417810_031/EMIT_L2A_MASK_001_20240626T152155_2417810_031.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152144_2417810_030/EMIT_L2A_RFL_001_20240626T152144_2417810_030.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152144_2417810_030/EMIT_L2A_RFLUNCERT_001_20240626T152144_2417810_030.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152144_2417810_030/EMIT_L2A_MASK_001_20240626T152144_2417810_030.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152132_2417810_029/EMIT_L2A_RFL_001_20240626T152132_2417810_029.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152132_2417810_029/EMIT_L2A_MASK_001_20240626T152132_2417810_029.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152132_2417810_029/EMIT_L2A_RFLUNCERT_001_20240626T152132_2417810_029.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152120_2417810_028/EMIT_L2A_RFL_001_20240626T152120_2417810_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152120_2417810_028/EMIT_L2A_RFLUNCERT_001_20240626T152120_2417810_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152120_2417810_028/EMIT_L2A_MASK_001_20240626T152120_2417810_028.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152108_2417810_027/EMIT_L2A_RFL_001_20240626T152108_2417810_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152108_2417810_027/EMIT_L2A_RFLUNCERT_001_20240626T152108_2417810_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152108_2417810_027/EMIT_L2A_MASK_001_20240626T152108_2417810_027.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152056_2417810_026/EMIT_L2A_RFL_001_20240626T152056_2417810_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152056_2417810_026/EMIT_L2A_RFLUNCERT_001_20240626T152056_2417810_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152056_2417810_026/EMIT_L2A_MASK_001_20240626T152056_2417810_026.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152044_2417810_025/EMIT_L2A_RFL_001_20240626T152044_2417810_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152044_2417810_025/EMIT_L2A_RFLUNCERT_001_20240626T152044_2417810_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152044_2417810_025/EMIT_L2A_MASK_001_20240626T152044_2417810_025.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152033_2417810_024/EMIT_L2A_RFL_001_20240626T152033_2417810_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152033_2417810_024/EMIT_L2A_RFLUNCERT_001_20240626T152033_2417810_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152033_2417810_024/EMIT_L2A_MASK_001_20240626T152033_2417810_024.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152021_2417810_023/EMIT_L2A_RFL_001_20240626T152021_2417810_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152021_2417810_023/EMIT_L2A_RFLUNCERT_001_20240626T152021_2417810_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240626T152021_2417810_023/EMIT_L2A_MASK_001_20240626T152021_2417810_023.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165922_2417411_056/EMIT_L2A_RFL_001_20240622T165922_2417411_056.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165922_2417411_056/EMIT_L2A_RFLUNCERT_001_20240622T165922_2417411_056.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165922_2417411_056/EMIT_L2A_MASK_001_20240622T165922_2417411_056.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165755_2417411_055/EMIT_L2A_RFL_001_20240622T165755_2417411_055.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165755_2417411_055/EMIT_L2A_RFLUNCERT_001_20240622T165755_2417411_055.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165755_2417411_055/EMIT_L2A_MASK_001_20240622T165755_2417411_055.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165743_2417411_054/EMIT_L2A_RFL_001_20240622T165743_2417411_054.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165743_2417411_054/EMIT_L2A_RFLUNCERT_001_20240622T165743_2417411_054.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165743_2417411_054/EMIT_L2A_MASK_001_20240622T165743_2417411_054.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165731_2417411_053/EMIT_L2A_RFL_001_20240622T165731_2417411_053.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165731_2417411_053/EMIT_L2A_RFLUNCERT_001_20240622T165731_2417411_053.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165731_2417411_053/EMIT_L2A_MASK_001_20240622T165731_2417411_053.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165719_2417411_052/EMIT_L2A_RFL_001_20240622T165719_2417411_052.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165719_2417411_052/EMIT_L2A_RFLUNCERT_001_20240622T165719_2417411_052.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165719_2417411_052/EMIT_L2A_MASK_001_20240622T165719_2417411_052.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165707_2417411_051/EMIT_L2A_RFL_001_20240622T165707_2417411_051.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165707_2417411_051/EMIT_L2A_MASK_001_20240622T165707_2417411_051.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165707_2417411_051/EMIT_L2A_RFLUNCERT_001_20240622T165707_2417411_051.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165655_2417411_050/EMIT_L2A_RFL_001_20240622T165655_2417411_050.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165655_2417411_050/EMIT_L2A_MASK_001_20240622T165655_2417411_050.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165655_2417411_050/EMIT_L2A_RFLUNCERT_001_20240622T165655_2417411_050.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165644_2417411_049/EMIT_L2A_RFL_001_20240622T165644_2417411_049.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165644_2417411_049/EMIT_L2A_RFLUNCERT_001_20240622T165644_2417411_049.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165644_2417411_049/EMIT_L2A_MASK_001_20240622T165644_2417411_049.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165632_2417411_048/EMIT_L2A_RFL_001_20240622T165632_2417411_048.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165632_2417411_048/EMIT_L2A_RFLUNCERT_001_20240622T165632_2417411_048.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165632_2417411_048/EMIT_L2A_MASK_001_20240622T165632_2417411_048.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165620_2417411_047/EMIT_L2A_RFL_001_20240622T165620_2417411_047.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165620_2417411_047/EMIT_L2A_RFLUNCERT_001_20240622T165620_2417411_047.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165620_2417411_047/EMIT_L2A_MASK_001_20240622T165620_2417411_047.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165608_2417411_046/EMIT_L2A_RFL_001_20240622T165608_2417411_046.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165608_2417411_046/EMIT_L2A_RFLUNCERT_001_20240622T165608_2417411_046.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240622T165608_2417411_046/EMIT_L2A_MASK_001_20240622T165608_2417411_046.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240621T174554_2417312_051/EMIT_L2A_RFL_001_20240621T174554_2417312_051.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240621T174554_2417312_051/EMIT_L2A_RFLUNCERT_001_20240621T174554_2417312_051.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240621T174554_2417312_051/EMIT_L2A_MASK_001_20240621T174554_2417312_051.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240621T174542_2417312_050/EMIT_L2A_RFL_001_20240621T174542_2417312_050.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240621T174542_2417312_050/EMIT_L2A_RFLUNCERT_001_20240621T174542_2417312_050.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240621T174542_2417312_050/EMIT_L2A_MASK_001_20240621T174542_2417312_050.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205905_2416314_010/EMIT_L2A_RFL_001_20240611T205905_2416314_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205905_2416314_010/EMIT_L2A_RFLUNCERT_001_20240611T205905_2416314_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205905_2416314_010/EMIT_L2A_MASK_001_20240611T205905_2416314_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205853_2416314_009/EMIT_L2A_RFL_001_20240611T205853_2416314_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205853_2416314_009/EMIT_L2A_MASK_001_20240611T205853_2416314_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205853_2416314_009/EMIT_L2A_RFLUNCERT_001_20240611T205853_2416314_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205842_2416314_008/EMIT_L2A_RFL_001_20240611T205842_2416314_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205842_2416314_008/EMIT_L2A_RFLUNCERT_001_20240611T205842_2416314_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205842_2416314_008/EMIT_L2A_MASK_001_20240611T205842_2416314_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205658_2416314_003/EMIT_L2A_RFL_001_20240611T205658_2416314_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205658_2416314_003/EMIT_L2A_RFLUNCERT_001_20240611T205658_2416314_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205658_2416314_003/EMIT_L2A_MASK_001_20240611T205658_2416314_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205646_2416314_002/EMIT_L2A_RFL_001_20240611T205646_2416314_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205646_2416314_002/EMIT_L2A_MASK_001_20240611T205646_2416314_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240611T205646_2416314_002/EMIT_L2A_RFLUNCERT_001_20240611T205646_2416314_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240610T214646_2416214_008/EMIT_L2A_RFL_001_20240610T214646_2416214_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240610T214646_2416214_008/EMIT_L2A_RFLUNCERT_001_20240610T214646_2416214_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240610T214646_2416214_008/EMIT_L2A_MASK_001_20240610T214646_2416214_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240610T214634_2416214_007/EMIT_L2A_RFL_001_20240610T214634_2416214_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240610T214634_2416214_007/EMIT_L2A_RFLUNCERT_001_20240610T214634_2416214_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240610T214634_2416214_007/EMIT_L2A_MASK_001_20240610T214634_2416214_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124755_2416109_004/EMIT_L2A_RFL_001_20240609T124755_2416109_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124755_2416109_004/EMIT_L2A_RFLUNCERT_001_20240609T124755_2416109_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124755_2416109_004/EMIT_L2A_MASK_001_20240609T124755_2416109_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124743_2416109_003/EMIT_L2A_RFL_001_20240609T124743_2416109_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124743_2416109_003/EMIT_L2A_RFLUNCERT_001_20240609T124743_2416109_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124743_2416109_003/EMIT_L2A_MASK_001_20240609T124743_2416109_003.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124731_2416109_002/EMIT_L2A_RFL_001_20240609T124731_2416109_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124731_2416109_002/EMIT_L2A_MASK_001_20240609T124731_2416109_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124731_2416109_002/EMIT_L2A_RFLUNCERT_001_20240609T124731_2416109_002.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124719_2416109_001/EMIT_L2A_RFL_001_20240609T124719_2416109_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124719_2416109_001/EMIT_L2A_RFLUNCERT_001_20240609T124719_2416109_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240609T124719_2416109_001/EMIT_L2A_MASK_001_20240609T124719_2416109_001.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133613_2416009_014/EMIT_L2A_RFL_001_20240608T133613_2416009_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133613_2416009_014/EMIT_L2A_RFLUNCERT_001_20240608T133613_2416009_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133613_2416009_014/EMIT_L2A_MASK_001_20240608T133613_2416009_014.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133602_2416009_013/EMIT_L2A_RFL_001_20240608T133602_2416009_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133602_2416009_013/EMIT_L2A_RFLUNCERT_001_20240608T133602_2416009_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133602_2416009_013/EMIT_L2A_MASK_001_20240608T133602_2416009_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133550_2416009_012/EMIT_L2A_RFL_001_20240608T133550_2416009_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133550_2416009_012/EMIT_L2A_RFLUNCERT_001_20240608T133550_2416009_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133550_2416009_012/EMIT_L2A_MASK_001_20240608T133550_2416009_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133538_2416009_011/EMIT_L2A_RFL_001_20240608T133538_2416009_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133538_2416009_011/EMIT_L2A_RFLUNCERT_001_20240608T133538_2416009_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133538_2416009_011/EMIT_L2A_MASK_001_20240608T133538_2416009_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133526_2416009_010/EMIT_L2A_RFL_001_20240608T133526_2416009_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133526_2416009_010/EMIT_L2A_RFLUNCERT_001_20240608T133526_2416009_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133526_2416009_010/EMIT_L2A_MASK_001_20240608T133526_2416009_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133514_2416009_009/EMIT_L2A_RFL_001_20240608T133514_2416009_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133514_2416009_009/EMIT_L2A_RFLUNCERT_001_20240608T133514_2416009_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133514_2416009_009/EMIT_L2A_MASK_001_20240608T133514_2416009_009.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133447_2416009_008/EMIT_L2A_RFL_001_20240608T133447_2416009_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133447_2416009_008/EMIT_L2A_RFLUNCERT_001_20240608T133447_2416009_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133447_2416009_008/EMIT_L2A_MASK_001_20240608T133447_2416009_008.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133435_2416009_007/EMIT_L2A_RFL_001_20240608T133435_2416009_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133435_2416009_007/EMIT_L2A_RFLUNCERT_001_20240608T133435_2416009_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133435_2416009_007/EMIT_L2A_MASK_001_20240608T133435_2416009_007.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133423_2416009_006/EMIT_L2A_RFL_001_20240608T133423_2416009_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133423_2416009_006/EMIT_L2A_RFLUNCERT_001_20240608T133423_2416009_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133423_2416009_006/EMIT_L2A_MASK_001_20240608T133423_2416009_006.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133411_2416009_005/EMIT_L2A_RFL_001_20240608T133411_2416009_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133411_2416009_005/EMIT_L2A_RFLUNCERT_001_20240608T133411_2416009_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133411_2416009_005/EMIT_L2A_MASK_001_20240608T133411_2416009_005.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133359_2416009_004/EMIT_L2A_RFL_001_20240608T133359_2416009_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133359_2416009_004/EMIT_L2A_RFLUNCERT_001_20240608T133359_2416009_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240608T133359_2416009_004/EMIT_L2A_MASK_001_20240608T133359_2416009_004.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133749_2415809_013/EMIT_L2A_RFL_001_20240606T133749_2415809_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133749_2415809_013/EMIT_L2A_RFLUNCERT_001_20240606T133749_2415809_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133749_2415809_013/EMIT_L2A_MASK_001_20240606T133749_2415809_013.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133737_2415809_012/EMIT_L2A_RFL_001_20240606T133737_2415809_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133737_2415809_012/EMIT_L2A_MASK_001_20240606T133737_2415809_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133737_2415809_012/EMIT_L2A_RFLUNCERT_001_20240606T133737_2415809_012.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133725_2415809_011/EMIT_L2A_RFL_001_20240606T133725_2415809_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133725_2415809_011/EMIT_L2A_RFLUNCERT_001_20240606T133725_2415809_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133725_2415809_011/EMIT_L2A_MASK_001_20240606T133725_2415809_011.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133713_2415809_010/EMIT_L2A_RFL_001_20240606T133713_2415809_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133713_2415809_010/EMIT_L2A_RFLUNCERT_001_20240606T133713_2415809_010.nc
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20240606T133713_2415809_010/EMIT_L2A_MASK_001_20240606T133713_2415809_010.nc
EDSCEOF
