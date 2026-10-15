from src.app.biograph_service import BioGraphService

svc = BioGraphService()
print([n for n in svc.drug_names() if "etformin" in n])

for row in svc.rank_diseases("Metformin", top_n=5):
    print(row)