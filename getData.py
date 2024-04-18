import extraction
import base64
import dataclasses
import datetime
import gzip
import hashlib
from typing import BinaryIO, Iterable, Iterator, Union
import json
import pytz

# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""WMT docs extraction.

The exported functions are:

`get_deduplicated_wmt_docs`
  1) takes the `gz` document-split versions of the WMT News Crawl from
     'https://data.statmt.org/news-crawl/README',
  2) extracts the documents,
  3) filters out duplicates,
  4) and then yields the lines parsed into `WMTDoc` objects.

`get_wmt_passages_from_docs`
  1) takes the `WMTDoc`s from the previous output
  2) splits the articles into sentences
  3) and yields them as `WMTPassage` chunks.
"""

import base64
import dataclasses
import datetime
import gzip
import hashlib
from typing import BinaryIO, Iterable, Iterator, Union

import pytz

_EXTRACTION_FIELD_SEPARATOR = b'\t'
_EXTRACTION_DATE_FORMAT = '%Y%m%d'

_SORTING_KEY_DATE_FORMAT = '%Y%m%d%H%M%S%f'
_SORTING_KEY_FIELD_SEPARATOR = '\x00\x01'

_PASSAGE_ID = '{sorting_key}_{passage_idx}'
_PASSAGE_SENTENCE_SEPARATOR = b'. '
_PASSAGE_NUM_SENTENCES = 6
_PASSAGE_DATE_PREFIX_FORMAT = '%A, %B %-d, %Y'


@dataclasses.dataclass(frozen=True)
class WMTDoc:
  """The input extracted from the WMT archive files.

  Attributes:
    sorting_key: The assigned sorting key to the document.
    publication_ts: Publication date of the document as UTC timestamp seconds.
    text: The processed document text / article.
  """
  sorting_key: str
  publication_ts: int
  text: bytes


@dataclasses.dataclass(frozen=True)
class WMTPassage:
  """A passage from a sequence of `WMTDoc` article sentence chunks.

  Attributes:
    id: Assigned ID consisting of the original `WMTDoc` `sorting_key` and an
      index for the passage position in the original article.
    text: A chunk from a sequence of sentences extracted from the `WMTDoc`
      article.
  """
  id: str
  text: bytes

def _extract_doc(doc_line):
  """Processes a doc line (= one line per doc) from the WMT dataset file."""

  publication_date, sentence_split_doc_line, unsplit_doc_line = (
      doc_line.strip().split(_EXTRACTION_FIELD_SEPARATOR))


  # pylint: disable=g-tzinfo-replace
  publication_dt = datetime.datetime.strptime(
      publication_date.decode(),
      _EXTRACTION_DATE_FORMAT).replace(tzinfo=pytz.UTC)
  line_hash = hashlib.sha256(unsplit_doc_line).hexdigest()
  sorting_key = _SORTING_KEY_FIELD_SEPARATOR.join([
      publication_dt.strftime(_SORTING_KEY_DATE_FORMAT),
      line_hash,
      '',
  ])

  return WMTDoc(
      sorting_key=sorting_key,
      publication_ts=int(publication_dt.timestamp()),
      text=base64.b64decode(sentence_split_doc_line),
  )

def get_deduplicated_wmt_docs(wmt_archive_files, deduplicated_sorting_keys_file):
  # print('yes')
  with gzip.open(deduplicated_sorting_keys_file) as f:
    sorting_keys = set(line.strip().decode() for line in f)

  for file_path_or_object in wmt_archive_files:
    with gzip.open(file_path_or_object) as f:
      for line in f:
        doc = _extract_doc(line)
        if doc.sorting_key in sorting_keys:
          return doc

# _archive_file_names = [
#   'downloadData/news-docs.2010.en.filtered.gz',
# ]

wmt_docs = extraction.get_deduplicated_wmt_docs(
  wmt_archive_files=[
  '/shared/nas2/yujiz/effiUpdating/streamingqa/downloadData/news-docs.2020.en.filtered.gz',
],
  deduplicated_sorting_keys_file='/shared/nas2/yujiz/effiUpdating/streamingqa/downloadData/wmt_sorting_key_ids.txt.gz',
)

# wmt_passages = extraction.get_wmt_passages_from_docs(
#   wmt_docs=wmt_docs,
#   preprend_date=True,
# )

all_wmt_doc_list = []
for each_doc in wmt_docs:
  all_wmt_doc_list.append(each_doc)

# Define the file paths for the QA dataset
_file_name_by_streamingqa_subset = {
    'train': '/shared/nas2/yujiz/effiUpdating/streamingqa/downloadData/streaminqa_train.jsonl.gz',
    'valid': '/shared/nas2/yujiz/effiUpdating/streamingqa/downloadData/streaminqa_valid.jsonl.gz',
    'eval': '/shared/nas2/yujiz/effiUpdating/streamingqa/downloadData/streaminqa_eval.jsonl.gz',
}

# Load the QA dataset
streamingqa_eval = []
with gzip.open(_file_name_by_streamingqa_subset['eval'], 'rb') as input_file:
    for line in input_file:
        streamingqa_eval.append(json.loads(line.decode()))




