from modules.data.data import Data
from modules.data.disc_to_cont import disc_to_cont
from modules.data.generate_data import get_data

try:
    data_obj = Data()
except Exception as e:
    print(e)
    get_data()
    data_obj = Data()
