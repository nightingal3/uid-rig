import pandas as pd
from g2p import make_g2p
from g2p_en import G2p as G2p_en
from g2pc import G2pC
import pdb

if __name__ == "__main__":
    df = pd.read_csv("./data/terms_1_to_100/english.csv", encoding="latin1")
    df_fr = pd.read_csv("./data/terms_1_to_100/french.csv", encoding="latin1")
    df_cn = pd.read_csv("./data/terms_1_to_100/chinese.csv", encoding="utf-8")

    transducer_fr = make_g2p("fra", "eng-arpabet")
    g2p_en = G2p_en()
    g2p_cn = G2pC()

    df["phonetic"] = df["Reading"].apply(lambda x: " ".join(g2p_en(" ".join(x.split("-")))))
    df.to_csv("./data/terms_1_to_100/english_phonetic.csv", index=False)


    df_fr["phonetic"] = df_fr["Reading"].apply(lambda x: transducer_fr(" ".join(x.split("-"))).output_string)
    df_fr.to_csv("./data/terms_1_to_100/french_phonetic.csv", index=False)

    df_cn["phonetic"] = df_cn["Reading"].apply(lambda x: g2p_cn(x)[2])
    df_cn.to_csv("./data/terms_1_to_100/french_phonetic.csv", index=False)
    