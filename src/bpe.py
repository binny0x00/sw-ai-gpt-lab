# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)} 
# SPECIAL_IDS 는 { "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, }를 의미함 - 특수 토큰 ID를 항상 고정하기 위해 사용하는 값
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
        self.merges = [] # 토크나이저가 학습한 merge 규칙 목록 (2개짜리 튜플을 원소로 가짐)

    def _init_special_tokens(self):
        """
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        # 0. id_to_token, token_to_id를 비운다.
        self.id_to_token = {}
        self.token_to_id = {}

        # 1. <pad>, <unk>, <bos>, <eos>를 ID 0~3으로 등록한다.
        for token, idx in SPECIAL_IDS.items():
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

        # 2. byte 0~255를 ID 4~259로 등록한다.
        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            token = bytes([byte_value]) # bytes()는 Python에서 변경 불가능한 byte 시퀀스를 만드는 함수
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

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """

        # 문자열을 UTF-8 bytes로 변환한 뒤 byte token ID 리스트를 만든다.
        text_bytes = text.encode("utf-8") # bytes 객체(iterable 객체)
        ids = [BYTE_OFFSET + b for b in text_bytes] # 현재 입력 문장을 token id로 바꾼 작업용 리스트

        # merge rule을 학습 순서대로 적용
        for pair in self.merges:
            new_id = self.token_to_id[pair]
            ids = self._apply_merge(ids, pair, new_id)

        # add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙인다.
        if add_bos_eos:
            ids.insert(0, self.get_bos_id())
            ids.append(self.get_eos_id())
        
        return ids

    # [헬퍼 함수] encode 및 train을 위한 머지용 함수: ids 안에서 특정 pair를 새 token id로 치환한다
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

    # [헬퍼 함수] decode용 재귀함수: token ID를 최종 byte 값들로 변환 
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

        return []