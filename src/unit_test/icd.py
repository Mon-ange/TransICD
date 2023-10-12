from data.icd import icd_utility


def test_index_icd_descriptions():
    indexed_icd = icd_utility.getIndexedICDDescriptions(7)
    print(indexed_icd)
