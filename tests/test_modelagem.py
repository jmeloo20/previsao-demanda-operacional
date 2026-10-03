import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelagem import intervalo_diferenca, metricas, preparar, raio_empirico


class TestDemanda(unittest.TestCase):
    def base(self):
        return pd.DataFrame({"dteday": pd.date_range("2011-01-01", periods=60), "cnt": np.arange(60) + 100, "holiday": 0, "workingday": 1})

    def test_alvo_atual_e_futuro_nao_afetam_atributos_atuais(self):
        original = self.base()
        anterior, _ = preparar(original)
        original.loc[40:, "cnt"] = 99999
        alterado, _ = preparar(original)
        pd.testing.assert_frame_equal(anterior.loc[:"2011-02-10"], alterado.loc[:"2011-02-10"])

    def test_lags_e_medias(self):
        x, y = preparar(self.base())
        self.assertEqual(x.iloc[0].lag_1, 127)
        self.assertEqual(x.iloc[0].lag_7, 121)
        self.assertEqual(x.iloc[0].media_7, np.mean(np.arange(121, 128)))
        self.assertEqual(y.iloc[0], 128)

    def test_lacuna_nao_vira_zero(self):
        with self.assertRaises(ValueError):
            preparar(self.base().drop(index=20))

    def test_proximo_dia_sem_alvo(self):
        dados = self.base()
        dados.loc[59, "cnt"] = np.nan
        x, y = preparar(dados, permitir_ultimo_sem_alvo=True)
        self.assertEqual(x.iloc[-1].lag_1, 158)
        self.assertTrue(pd.isna(y.iloc[-1]))
        dados.loc[30, "cnt"] = np.nan
        with self.assertRaises(ValueError):
            preparar(dados, permitir_ultimo_sem_alvo=True)

    def test_metricas_e_intervalo(self):
        self.assertEqual(metricas([10, 20], [12, 18])["mae"], 2)
        self.assertAlmostEqual(metricas([10, 20], [12, 18])["wape"], 4 / 30)
        self.assertEqual(raio_empirico(np.arange(1, 10)), 9)
        np.testing.assert_array_equal(intervalo_diferenca(np.ones(20), np.ones(20), np.ones(20)), [0, 0])


if __name__ == "__main__":
    unittest.main()
