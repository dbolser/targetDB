import os
import sqlite3
import pandas as pd
import pytest

# Ensure package modules are importable when tests are executed from the tests directory
import sys
PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PACKAGE_ROOT)
sys.path.append(os.path.join(PACKAGE_ROOT, "targetDB"))
from targetDB import target_features


def build_test_db():
    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()
    cur.executescript('''
        CREATE TABLE Targets (Target_id TEXT PRIMARY KEY, Gene_name TEXT);
        INSERT INTO Targets VALUES ('T1','GENE1');

        CREATE TABLE disease (Target_id TEXT, disease_name TEXT, disease_id TEXT);
        INSERT INTO disease VALUES ('T1','DiseaseX','D1');

        CREATE TABLE pathways (Target_id TEXT, pathway_name TEXT, pathway_dataset TEXT);
        INSERT INTO pathways VALUES ('T1','ReactomePath','Reactome pathways data set');
        INSERT INTO pathways VALUES ('T1','KEGGPath','KEGG pathways data set');

        CREATE TABLE diff_exp_disease (Target_id TEXT, disease TEXT, t_stat REAL, expression_status TEXT);
        INSERT INTO diff_exp_disease VALUES ('T1','DiseaseX',2.0,'UP');

        CREATE TABLE gwas (Target_id TEXT, phenotype TEXT, organism TEXT, p_value REAL, first_author TEXT, publication_year INT, pubmed_id TEXT);
        INSERT INTO gwas VALUES ('T1','Phen1','Human',0.01,'Smith',2020,'PMID1');

        CREATE TABLE diff_exp_tissue (Target_id TEXT, Tissue TEXT, t_stat REAL);
        INSERT INTO diff_exp_tissue VALUES ('T1','Liver',1.5);

        CREATE TABLE protein_expression_selectivity (Target_id TEXT, Selectivity_entropy REAL);
        INSERT INTO protein_expression_selectivity VALUES ('T1',0.5);

        CREATE TABLE protein_expression_levels (Target_id TEXT, organ TEXT, tissue TEXT, cell TEXT, value REAL);
        INSERT INTO protein_expression_levels VALUES ('T1','Organ1','Tissue1','Cell1',1.0);

        CREATE TABLE phenotype (Target_id TEXT, Allele_id INT, Allele_symbol TEXT, Allele_type TEXT, zygosity TEXT, genotype TEXT, Phenotype TEXT);
        INSERT INTO phenotype VALUES ('T1',1,'ALLELE1','type1','het','geno1','Phenotype1');

        CREATE TABLE Isoforms (Target_id TEXT, Isoform_id TEXT, Isoform_name TEXT, Sequence TEXT, n_residues INT, Canonical INT, Identity REAL);
        INSERT INTO Isoforms VALUES ('T1','ISO1','1','SEQ',100,1,99.0);

        CREATE TABLE isoform_modifications (isoform_id TEXT, mod_id TEXT);
        INSERT INTO isoform_modifications VALUES ('ISO1','MOD1');

        CREATE TABLE modifications (Unique_modID TEXT, Target_id TEXT, start INT, stop INT, previous TEXT, action TEXT, new TEXT, domains TEXT, comment TEXT, mod_type TEXT);
        INSERT INTO modifications VALUES ('MOD1','T1',1,2,'A','del','B','domain1','comment1','MOD');
        INSERT INTO modifications VALUES ('VAR1','T1',3,4,'C','sub','D','domain2','comment2','VAR');
        INSERT INTO modifications VALUES ('MUT1','T1',5,6,'E','add','F','domain3','comment3','MUTAGEN');

        CREATE TABLE Domain_targets (domain_id INT, Target_id TEXT, Domain_name TEXT, Domain_start INT, Domain_stop INT, length INT, source_name TEXT);
        INSERT INTO Domain_targets VALUES (1,'T1','Domain1',1,50,50,'Source1');

        CREATE TABLE "3D_Blast" (Query_target_id TEXT, Hit_PDB_code TEXT, Chain_Letter TEXT, similarity REAL, Hit_gene_name TEXT, Hit_gene_species TEXT);
        INSERT INTO "3D_Blast" VALUES ('T1','PDB1','A',0.9,'GeneP','SpeciesP');

        CREATE TABLE drugEbility_sites (pdb_code TEXT, tractable INT, druggable INT);
        INSERT INTO drugEbility_sites VALUES ('PDB1',1,1);
        INSERT INTO drugEbility_sites VALUES ('pdb1',1,1);

        CREATE TABLE PDB_Chains (Chain_id INT, PDB_code TEXT, Chain TEXT, n_residues INT, start_stop TEXT, Target_id TEXT);
        INSERT INTO PDB_Chains VALUES (1,'PDB1','A',100,'1-100','T1');

        CREATE TABLE PDB (PDB_code TEXT, Technique TEXT, Resolution REAL);
        INSERT INTO PDB VALUES ('PDB1','X-ray',1.2);

        CREATE TABLE PDBChain_Domain (Chain_id INT, Domain_id INT);
        INSERT INTO PDBChain_Domain VALUES (1,1);

        CREATE TABLE pdb_bind (pdb_code TEXT, type TEXT, binding_type TEXT, binding_operator TEXT, binding_value REAL, binding_units TEXT, lig_name TEXT, pub_year INT);
        INSERT INTO pdb_bind VALUES ('PDB1','type1','bind','=',10,'nM','Lig1',2021);

        CREATE TABLE fPockets (Pocket_id INT, Target_id TEXT, PDB_code TEXT, DrugScore REAL, total_sasa REAL, volume REAL, apolar_sasa REAL, Pocket_number INT, Score REAL, druggable TEXT, blast TEXT);
        INSERT INTO fPockets VALUES (1,'T1','PDB1',0.8,100.0,50.0,20.0,1,5.0,'TRUE','FALSE');
        INSERT INTO fPockets VALUES (2,'T1','PDB1',0.7,90.0,40.0,15.0,2,4.0,'TRUE','TRUE');

        CREATE TABLE fPockets_Domain (Pocket_id INT, Domain_id INT, Coverage INT);
        INSERT INTO fPockets_Domain VALUES (1,1,80);

        CREATE TABLE Crossref (target_id TEXT, Chembl_id TEXT);
        INSERT INTO Crossref VALUES ('T1','CHEMBL1');

        CREATE TABLE bioactivities (
            lig_id TEXT, assay_id TEXT, Target_id TEXT, target_name TEXT,
            standard_type TEXT, operator TEXT, value_num REAL, units TEXT,
            activity_comment TEXT, data_validity_comment TEXT, doi TEXT,
            pchembl_value REAL, bioactivity_type TEXT, assay_species TEXT,
            assay_description TEXT, confidence_score INT, SMILES TEXT,
            HBA INT, HBD INT, LogD REAL, LogP REAL, MW REAL, TPSA REAL,
            aLogP REAL, apKa REAL, bpKa REAL, nAr INT, pass_ro3 TEXT,
            ro5_violations INT, rotB INT, mol_name TEXT, molecular_species TEXT,
            indication_class TEXT, class_def TEXT, max_phase INT, oral TEXT,
            assay_ref TEXT, ref_bio TEXT
        );
        INSERT INTO bioactivities VALUES (
            'L1','A1','CHEMBL1','TargetName','Ki','=',10,'nM','',NULL,'doi',7.0,
            'Binding','Human','assaydesc',8,'C',1,1,1,1,1,1,1,1,1,1,'Yes',0,0,'Ligand1','species','indication','class',1,'Yes','assayref','ref');

        CREATE TABLE ligands (
            lig_id TEXT PRIMARY KEY, mol_name TEXT, max_phase INT, oral TEXT,
            indication_class TEXT, class_def TEXT, alogp REAL, acd_logd REAL,
            acd_logp REAL, acd_most_apka REAL, acd_most_bpka REAL, HBA INT,
            HBD INT, TPSA REAL, molecularWeight REAL, rotatableBonds INT,
            n_Ar_rings INT, molecular_species TEXT, num_ro5_violations INT,
            ro3_pass TEXT, canonical_smiles TEXT, std_inchi_key TEXT
        );
        INSERT INTO ligands VALUES ('L1','Ligand1',1,'Yes','indication','class',1,1,1,1,1,1,1,1,1,1,1,'species',0,'Yes','C','KEY1');

        CREATE TABLE assays (
            assay_id TEXT, assay_description TEXT, species TEXT,
            bioactivity_type TEXT, confidence_score INT, doi TEXT
        );
        INSERT INTO assays VALUES ('A1','assaydesc','Human','Binding',8,'assaydoi');

        CREATE TABLE purchasable_compounds (target_id TEXT, smiles TEXT, affinity_type TEXT, affinity_value REAL, affinity_unit TEXT, price REAL, website TEXT);
        INSERT INTO purchasable_compounds VALUES ('T1','C','IC50',5,'nM',100,'http://example.com');

        CREATE TABLE BindingDB (
            target_id TEXT, ligand_name TEXT, ZincID TEXT,
            `IC50(nM)` REAL, `EC50(nM)` REAL, `Ki(nM)` REAL, `Kd(nM)` REAL,
            `kon(M-1s-1)` REAL, `koff(s-1)` REAL, pH REAL, `Temp` REAL,
            Source TEXT, DOI TEXT, institution TEXT, patent_number TEXT,
            ligand_smiles TEXT, inchi_key TEXT
        );
        INSERT INTO BindingDB VALUES ('T1','LigandBD','Z1',1,2,3,4,5,6,7,8,'source','doi','inst','patent','C','KEY1');

        CREATE TABLE drugEbility_domains (pdb_code TEXT, domain_fold TEXT, domain_superfamily TEXT, tractable INT, druggable INT);
        INSERT INTO drugEbility_domains VALUES ('pdb1','fold1','superfamily1',1,1);

        CREATE TABLE opentarget_association (target_id TEXT, association_score REAL);
        INSERT INTO opentarget_association VALUES ('T1',0.1);
    ''')
    conn.commit()
    return conn


def test_get_single_features(monkeypatch):
    conn = build_test_db()
    monkeypatch.setattr(target_features.sqlite3, 'connect', lambda _: conn)

    results = target_features.get_single_features('T1', dbase=':memory:')

    assert list(results['general_info'].columns) == ['Target_id','Gene_name']
    assert results['general_info'].iloc[0]['Gene_name'] == 'GENE1'

    assert list(results['disease'].columns) == ['disease_name','disease_id']
    assert results['disease'].iloc[0]['disease_name'] == 'DiseaseX'

    assert results['reactome'].iloc[0]['pathway_name'] == 'ReactomePath'
    assert results['kegg'].iloc[0]['pathway_name'] == 'KEGGPath'

    assert results['disease_exp'].iloc[0]['disease'] == 'DiseaseX'
    assert pytest.approx(results['disease_exp'].iloc[0]['t_stat']) == 2.0

    assert results['gwas'].iloc[0]['phenotype'] == 'Phen1'

    assert results['tissue'].iloc[0]['Tissue'] == 'Liver'

    assert results['selectivity'].iloc[0]['Selectivity_entropy'] == 0.5

    assert results['organ_expression'].iloc[0]['organ_name'] == 'Organ1'

    assert results['tissue_expression'].iloc[0]['tissue'] == 'Tissue1'

    assert results['phenotype'].iloc[0]['zygosity'] == 'HET'

    assert results['isoforms'].iloc[0]['isoform_name'] == 'GENE1-1'

    assert results['isoforms_mod'].iloc[0]['modification_type'] == 'del'
    assert results['var'].iloc[0]['modification_type'] == 'sub'
    assert results['mut'].iloc[0]['modification_type'] == 'add'

    assert results['domains'].iloc[0]['Domain_name'] == 'Domain1'

    assert results['pdb_blast'].iloc[0]['PDB_code'] == 'PDB1'
    assert results['pdb'].iloc[0]['PDB_code'] == 'PDB1'

    assert results['pockets'].iloc[0]['pocket_number'] == 1
    assert results['alt_pockets'].iloc[0]['pocket_number'] == 2

    assert results['bioactives'].iloc[0]['lig_id'] == 'L1'

    assert results['commercials'].iloc[0]['smiles'] == 'C'

    assert results['bindingDB'].iloc[0]['ligand_name'] == 'LigandBD'

    assert results['domain_drugE'].iloc[0]['domain_fold'] == 'fold1'

    assert results['open_target'].iloc[0]['association_score'] == 0.1
