#!/bin/bash

read -r VERSION < "$(dirname $0)/DUMP_VERSION"
LINK="https://dumps.wikimedia.org/enwiki/${VERSION}/enwiki-${VERSION}-pages-articles.xml.bz2"

cmd="wget -c -np -A 7z ${LINK} -O enwiki-${VERSION}-pages-articles.xml.bz2"

echo "${LINK}"

${cmd}
