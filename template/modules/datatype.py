from dataclasses import dataclass 
import pandas as pd 
import numpy as np 

@dataclass
class Signal:
    barcode: str
    signal: np.array
    date: str

    @property 
    def area(self) -> float:
        return np.sum(self.signal)
        

@dataclass
class Vaccum(Signal):
    pass


@dataclass
class Charging(Signal):
    pass