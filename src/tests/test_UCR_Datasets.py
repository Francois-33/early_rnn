import sys
sys.path.append("..")

import unittest
from datasets.UCR_Dataset import UCRDataset

class TestTune(unittest.TestCase):

    def test_UCR_Dataset(self):

        trace_dataset = UCRDataset("Trace")
        self.assertTrue(trace_dataset.ndims,1)

        available_datasets = trace_dataset.dataset.list_datasets()

        expected_available_datasets = ['Adiac',
             'ArrowHead',
             'Beef',
             'BeetleFly',
             'BirdChicken',
             'Car',
             'CBF',
             'ChlorineConcentration',
             'CinCECGTorso',
             'Coffee',
             'Computers',
             'CricketX',
             'CricketY',
             'CricketZ',
             'DiatomSizeReduction',
             'DistalPhalanxOutlineCorrect',
             'DistalPhalanxOutlineAgeGroup',
             'DistalPhalanxTW',
             'Earthquakes',
             'ECG200',
             'ECG5000',
             'ECGFiveDays',
             'ElectricDevices',
             'FaceAll',
             'FaceFour',
             'FacesUCR',
             'FiftyWords',
             'Fish',
             'FordA',
             'FordB',
             'GunPoint',
             'Ham',
             'HandOutlines',
             'Haptics',
             'Herring',
             'InlineSkate',
             'InsectWingbeatSound',
             'ItalyPowerDemand',
             'LargeKitchenAppliances',
             'Lightning2',
             'Lightning7',
             'Mallat',
             'Meat',
             'MedicalImages',
             'MiddlePhalanxOutlineCorrect',
             'MiddlePhalanxOutlineAgeGroup',
             'MiddlePhalanxTW',
             'MoteStrain',
             'NonInvasiveFatalECGThorax1',
             'NonInvasiveFatalECGThorax2',
             'OliveOil',
             'OSULeaf',
             'PhalangesOutlinesCorrect',
             'Phoneme',
             'Plane',
             'ProximalPhalanxOutlineCorrect',
             'ProximalPhalanxOutlineAgeGroup',
             'ProximalPhalanxTW',
             'RefrigerationDevices',
             'ScreenType',
             'ShapeletSim',
             'ShapesAll',
             'SmallKitchenAppliances',
             'SonyAIBORobotSurface1',
             'SonyAIBORobotSurface2',
             'StarLightCurves',
             'Strawberry',
             'SwedishLeaf',
             'Symbols',
             'SyntheticControl',
             'ToeSegmentation1',
             'ToeSegmentation2',
             'Trace',
             'TwoLeadECG',
             'TwoPatterns',
             'UWaveGestureLibraryX',
             'UWaveGestureLibraryY',
             'UWaveGestureLibraryZ',
             'UWaveGestureLibraryAll',
             'Wafer',
             'Wine',
             'WordSynonyms',
             'Worms',
             'WormsTwoClass',
             'Yoga']

        self.assertListEqual(available_datasets, expected_available_datasets)

    def test_binary_dataset(self):
        dataset = UCRDataset("Wafer")
        self.assertEqual(dataset.y.min(), 0)
        self.assertEqual(dataset.y.max(), 1)
        self.assertEqual(dataset.nclasses, 2)

    def test_dataset_start_zero(self):
         dataset = UCRDataset("Lightning7")
         self.assertEqual(dataset.y.min(), 0)
         self.assertEqual(dataset.y.max(), 6)
         self.assertEqual(dataset.nclasses, 7)

if __name__ == '__main__':
    unittest.main()