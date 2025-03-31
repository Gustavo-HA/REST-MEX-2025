TRAIN_FILE = "../data/Rest-Mex_2025_train.csv"
TEST_FILE = None # Aún no liberan el test set

# Targets
TARGET1 = "Polarity"
TARGET2 = "Town"
TARGET3 = "Type"
TARGETS = [TARGET1, TARGET2, TARGET3]


# Columnas texto
TEXT_COLUMNS = ["Title", "Review"]

# Columna resultado
NEW_COLUMN = "Texto_Limpio"