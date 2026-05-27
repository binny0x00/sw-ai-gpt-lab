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
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
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
        self.id_to_token = {}
        self.token_to_id = {}
        # 토크나이저가 학습한 merge 규칙 목록 (2개짜리 튜플을 원소로 가짐)
        self.merges = []

    def _init_special_tokens(self):
        """
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        self.id_to_token = {}
        self.token_to_id = {}

        # 1. <pad>, <unk>, <bos>, <eos>를 ID 0~3으로 등록
        for token, idx in SPECIAL_IDS.items():
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

        # 2. byte 0~255를 ID 4~259로 등록
        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
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

    # BPE는 자주 등장하는 인접 token pair를 반복적으로 merge해 vocab을 확장
    def train(self, corpus: str):
        """
        코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """

        # 학습 시작 시 vocab과 merge rule을 초기화
        self._init_special_tokens()
        self.merges = []

        # corpus를 byte token id 리스트로 변환
        ids = [BYTE_OFFSET + b for b in corpus.encode("utf-8")]

        # 목표 vocab 크기까지 pair merge를 반복
        while len(self.id_to_token) < self.vocab_size:
            # 현재 ids에서 인접 pair 빈도 계산
            pair_counts = self._get_pair_counts(ids)
            if not pair_counts: # pair가 없으면 정지
                break

            # 가장 자주 나온 pair 선택
            best_pair = max(pair_counts, key=pair_counts.get)

            # 새 token id 발급
            new_id = len(self.id_to_token)

            # merges, id_to_token, token_to_id에 등록
            self.merges.append(best_pair)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id

            # ids에서 해당 pair를 새 token id로 치환
            ids = self._apply_merge(ids, best_pair, new_id)

    # [헬퍼 함수] 인접 token pair별 등장 횟수를 세는 함수
    def _get_pair_counts(self, ids: list[int]) -> dict[tuple[int, int], int]:
        counts = {} # pair 빈도표 dict
        
        for i in range(len(ids) - 1):
            # 현재 토큰과 바로 다음 토큰을 2개짜리 tuple로 묶는다.
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def save(self, path: str | Path):
        """
        vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        # (참고) token_to_id 는 load()에서 다시 만들 수 있으므로 저장할 필요 없음
        id_to_token_data = {}
        
        # id_to_token의 bytes/tuple 값을 JSON으로 저장할 수 있는 형태로 변환
        for token_id, token in self.id_to_token.items():
            if isinstance(token, str): # 특수 토큰
                item = {"type": "str", "value": token}
            elif isinstance(token, bytes): # 일반 토큰
                item = {"type": "bytes", "value": list(token)}
            elif isinstance(token, tuple): # 병합 토큰
                item = {"type": "tuple", "value": list(token)}
            else:
                raise ValueError(f"Unknown token type: {type(token)}")

            id_to_token_data[str(token_id)] = item

        data = {
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges], # tuple을 list로 변환
            "id_to_token": id_to_token_data,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2) 
            # ensure_ascii=False: 한글 같은 non-ASCII 문자를 그대로 저장
            # indent=2: JSON을 2칸 들여쓰기해서 저장하라는 뜻

    def load(self, path: str | Path):
        """
        save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        # JSON에는 tuple이 없기 때문에 load시 다시 tuple로 변환 필요
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.id_to_token = {}
        self.token_to_id = {}

        # JSON에는 bytes, tuple 타입이 없어서 다시 원래 타입으로 변환해야 함
        for token_id_str, item in data["id_to_token"].items():
            token_id = int(token_id_str) # JSON에 object key가 문자열로 저장되어서 token_id를 int로 복원

            if item["type"] == "str":
                token = item["value"] # 특수 토큰은 그대로 문자열로 반영
            elif item["type"] == "bytes":
                token = bytes(item["value"])
            elif item["type"] == "tuple":
                token = tuple(item["value"])
            else:
                raise ValueError(f"Unknown token type: {item['type']}")

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """

        # 문자열을 UTF-8 bytes로 변환한 뒤 byte token ID 리스트를 만든다.
        text_bytes = text.encode("utf-8")
        ids = [BYTE_OFFSET + b for b in text_bytes]

        # merge rule을 학습 순서대로 적용
        for pair in self.merges:
            new_id = self.token_to_id[pair]
            ids = self._apply_merge(ids, pair, new_id)

        # add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙인다.
        if add_bos_eos:
            ids.insert(0, self.get_bos_id())
            ids.append(self.get_eos_id())
        
        return ids

    # [헬퍼 함수] ids 안의 특정 pair를 새 token id로 치환 (merge시 사용)
    def _apply_merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged = []
        i = 0

        while i < len(ids):
            # 현재 원소 및 바로 다음 원소가 존재하고, 페어가 될 경우 머지
            if (i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair):
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1

        return merged

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """

        byte_values = []

        for token_id in ids:
            # 특수 토큰은 문자열 결과에서 제외
            if skip_special and token_id in SPECIAL_IDS.values():
                continue
        
            byte_token = self._token_id_to_bytes(token_id)
            byte_values.extend(byte_token)

        return bytes(byte_values).decode("utf-8")

    # [헬퍼 함수] merge token을 재귀적으로 펼쳐 byte 값으로 복원 (decode용)
    def _token_id_to_bytes(self, token_id: int) -> list[int]:
        token = self.id_to_token[token_id]

        # token이 bytes 객체일 경우 그대로 반환
        if isinstance(token, bytes):
            return list(token)

        # merge token: 머지가 계속 이어질 수 있으므로 재귀적으로 처리
        if isinstance(token, tuple):
            byte_values = []
            for child_id in token:
                byte_token = self._token_id_to_bytes(child_id)
                byte_values.extend(byte_token)
            return byte_values
        
        # 특수 토큰처리 (skip_special=False일 경우)
        if isinstance(token, str) and token_id in SPECIAL_IDS.values():
            return list(token.encode("utf-8"))

        raise ValueError(f"Unsupported token type: {type(token)}")