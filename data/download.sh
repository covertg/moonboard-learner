#!/bin/bash
set -ue
cd "${0%/*}"  # https://stackoverflow.com/questions/6393551/what-is-the-meaning-of-0-in-a-bash-script

pages=2069  # Max number of pages for 2016 Moonboard as of 05/17/2020
output="raw_probs.out"

if [ -e ${output} ]; then
    echo "File ${output} already exists! Aborting to avoid an overwrite." >&2
    echo "# Lines, # Bytes: $(cat ${output} | wc -lc)" >&2
    exit 1
fi

# What follows is an outline of the structure of a cURL command which could scrape data from Moonboard, but not the full command.
# The full request will require cookie data and other identifiers from the user, after logging on to Moonboard.
# A little exploration in Firefox or Chrome DevTools should get you far!
do_post() {
    page=$1
    echo "POSTing for page ${page}..."    
    # The only change we need to make to each cURL command is page number
    data1="sort=&page=${page}&"
    data2=$'rest_of_the_payload'
    data="${data1}${data2}"
    # cURL into output file
    curl 'moonboard_url' \
    -H 'your_request_headers_here' \
    -H 'dont_forget_about_all_the_cookies' \
    --data-raw ${data} -s -S >> ${output}
    # Add newline for each page/separate json response
    echo '' >> ${output}
}

echo "We'll try to download ${pages} pages into the file ${output} as a multi-record json. (Each record, i.e. page, is one line.)"
for ((i = $pages; i > 0; i--)); do
    do_post $i
done