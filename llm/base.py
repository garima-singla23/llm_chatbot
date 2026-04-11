from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Union

class BaseLLM(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        pass