from modules.data.disc_to_cont import disc_to_cont
from modules.data.data import Data

try:
    from .generate_data import get_data
except Exception as e:
    print(e)

try:
    data_obj = Data()
except Exception as e:
    print(e)
