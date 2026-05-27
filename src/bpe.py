# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

import json
from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]   
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}  # enumberate: 리스트를 순회하면서 (인덱스, 값) 꺼냄
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}   # id -> token
        self.token_to_id = {}   # token -> id
        self.merges = []

    def _init_special_tokens(self):
        # 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        self.id_to_token = {idx: token for idx, token in enumerate(SPECIAL_TOKENS)}
        self.token_to_id = SPECIAL_IDS.copy()

        # byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        for byte_value in range(NUM_BYTES):
            token_id = byte_value + BYTE_OFFSET
            token = bytes([byte_value])
            
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    # 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.
    def train(self, corpus: str):
        """
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        # 초기화
        self._init_special_tokens()
        self.merges = []

        # `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        corpus_byte = corpus.encode("utf-8")

        ids = []
        for byte in corpus_byte:
            ids.append(self.token_to_id[bytes([byte])])

        # 가장 자주 등장하는 이웃 token pair를 찾습니다.
        # 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        while len(self.id_to_token) < self.vocab_size:
            pair_counts = {}

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                
                if pair in pair_counts:
                    pair_counts[pair] += 1
                else:
                    pair_counts[pair] = 1

            if pair_counts:
                best_pair = max(pair_counts, key=pair_counts.get)
            else:
                break

            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            self.merges.append(best_pair)

            new_ids = []
            i = 0
            while i < len(ids)-1:
                if (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
    
            if i < len(ids):
                new_ids.append(ids[i])

            ids = new_ids

    # vocabulary와 merge rule을 JSON 파일로 저장합니다.
    def save(self, path: str | Path):
        # bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.

        # JSON 형식에 맞게 변환
        id_to_token_save = {}
        for idx, item in self.id_to_token.items():

            if isinstance(item, str):
                id_to_token_save[idx] = {"type": "str", "value": item}
                continue
                
            if isinstance(item, bytes):
                id_to_token_save[idx] = {"type": "bytes", "value": list(item)}
                continue

            if isinstance(item, tuple):
                id_to_token_save[idx] = {"type": "tuple", "value": list(item)}
                continue

        merges_save = []
        for item in self.merges:
            merges_save.append(list(item))

        # 저장할 데이터 생성
        data = {
            'vocab_size': self.vocab_size,
            'id_to_token': id_to_token_save,
            'merges': merges_save
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
    def load(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        id_to_token_load = {}
        for idx, item in data['id_to_token'].items():
            idx = int(idx)
            if item['type'] == 'str':
                id_to_token_load[idx] = item['value']
                continue
                
            if item['type'] == 'bytes':
                id_to_token_load[idx] = bytes(item['value'])
                continue

            if item['type'] == 'tuple':
                id_to_token_load[idx] = tuple(item['value'])
                continue

        token_to_id_load = {}
        for idx, item in id_to_token_load.items():
            token_to_id_load[item] = idx

        self.vocab_size = data['vocab_size']
        self.merges = [tuple(pair) for pair in data['merges']]
        self.id_to_token = id_to_token_load
        self.token_to_id = token_to_id_load

    # 문자열을 token ID 리스트로 변환합니다.
    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        # UTF-8 byte ID 리스트를 만듭니다.
        byte_list = text.encode("utf-8")

        result = []
        for byte in byte_list:
            result.append(self.token_to_id[bytes([byte])])

        # train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        for pair in self.merges:
            new_id = self.token_to_id[pair]
            new_result = []

            i = 0
            while i < len(result):
                if i < len(result) - 1 and (result[i], result[i + 1]) == pair:
                    new_result.append(new_id)
                    i += 2
                else:
                    new_result.append(result[i])
                    i += 1
            
            result = new_result

        # add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        if add_bos_eos:
            result.insert(0, self.get_bos_id())
            result.append(self.get_eos_id())

        return result
        
    # token ID 리스트를 문자열로 복원합니다.
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        result = []
        byte_buff = []
        
        for token_id in ids:
            token = self.id_to_token[token_id]

            if isinstance(token, str):  # special 토큰
                # byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
                if len(byte_buff):
                    word = b''.join(byte_buff).decode("utf-8")
                    result.append(word)
                    byte_buff.clear()

                if not skip_special:
                    result.append(token)
                else:
                    continue

            #  merge token은 원본 byte token까지 재귀적으로 펼칩니다.
            if isinstance(token,(bytes,tuple)):
                byte_buff += self.decode_byte_helper(token_id)
                continue

        if len(byte_buff):
            word = b''.join(byte_buff).decode("utf-8")
            result.append(word)
            byte_buff.clear()

        result = ''.join(result)

        return result
    
    def decode_byte_helper(self, token_id):
        token = self.id_to_token[token_id]

        result = []

        if isinstance(token, bytes):
            result.append(token)
        
        if isinstance(token, tuple):    # merge 토큰
            result += self.decode_byte_helper(token[0])
            result += self.decode_byte_helper(token[1])

        return result