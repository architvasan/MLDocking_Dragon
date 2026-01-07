"""This module contains functions docking a molecule to a receptor using Openeye.

The code is adapted from this repository: https://github.com/inspiremd/Model-generation
"""
import os
if not os.getenv("DOCKING_SIM_DUMMY"):
    import MDAnalysis as mda
    from openeye import oechem, oedocking, oeomega
    from rdkit import Chem
    from rdkit.Chem import AllChem

import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import time
from time import perf_counter, sleep
import random
from functools import cache
import socket
import psutil
import logging
from threading import BrokenBarrierError

import dragon
import multiprocessing as mp
from dragon.native.process import current as current_process
from dragon.native.process_group import ProcessGroup
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.infrastructure.connection import Connection
from dragon.data.ddict import DDict
from dragon.infrastructure.policy import Policy
from dragon.native.machine import Node
from inference.run_inference import split_dict_keys
from dragon.utils import host_id

from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union

import yaml
from pydantic_settings import BaseSettings as _BaseSettings
from pydantic import validator

from logging_config import sim_logger as logger
from logging_config import setup_logger, stdout_to_logger

_T = TypeVar("_T")

PathLike = Union[Path, str]


def exception_handler(default_return: Any = None):
    """Handle exceptions in a function by returning a `default_return` value."""

    def decorator(func):
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(
                    f"{func.__name__} raised an exception: {e} "
                    f"On input {args}, {kwargs}\nReturning {default_return}"
                )
                return default_return

        return wrapper

    return decorator


def _resolve_path_exists(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    p = value.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _resolve_mkdir(value: Path) -> Path:
    p = value.resolve()
    p.mkdir(exist_ok=True, parents=True)
    return p


def path_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_path_exists)
    return _validator


def mkdir_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_mkdir)
    return _validator


class BaseSettings(_BaseSettings):
    """Base settings to provide an easier interface to read/write YAML files."""

    def dump_yaml(self, filename: PathLike) -> None:
        with open(filename, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)  # type: ignore

'''
Functions
'''

def smi_to_structure(smiles: str, output_file: Path, forcefield: str = "mmff") -> None:
    """Convert a SMILES file to a structure file.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    output_file : Path
        EIther an output PDB file or output SDF file.
    forcefield : str, optional
        Forcefield to use for 3D conformation generation
        (either "mmff" or "etkdg"), by default "mmff".
    """
    # Convert SMILES to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)

    # Generate a 3D conformation for the molecule
    if forcefield == "mmff":
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
    elif forcefield == "etkdg":
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    else:
        raise ValueError(f"Unknown forcefield: {forcefield}")

    # Write the molecule to a file
    if output_file.suffix == ".pdb":
        writer = Chem.PDBWriter(str(output_file))
    elif output_file.suffix == ".sdf":
        writer = Chem.SDWriter(str(output_file))
    else:
        raise ValueError(f"Invalid output file extension: {output_file}")
    writer.write(mol)
    writer.close()



def from_mol(mol, isomer=True, num_enantiomers=1):
    """
    Generates a set of conformers as an OEMol object
    Inputs:
        mol is an OEMol
        isomers is a boolean controling whether or not the various diasteriomers of a molecule are created
        num_enantiomers is the allowable number of enantiomers. For all, set to -1
    """
    # Turn off the GPU for omega
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.GetTorDriveOptions().SetUseGPU(False)
    omega = oeomega.OEOmega(omegaOpts)

    out_conf = []
    if not isomer:
        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            out_conf.append(mol)
        else:
            oechem.OEThrow.Warning(
                "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
            )

    elif isomer:
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), 12, True):
            enantiomer = oechem.OEMol(enantiomer)
            ret_code = omega.Build(enantiomer)
            if ret_code == oeomega.OEOmegaReturnCode_Success:
                out_conf.append(enantiomer)
                num_enantiomers -= 1
                if num_enantiomers == 0:
                    break
            else:
                oechem.OEThrow.Warning(
                    "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
                )
    return out_conf


def from_string(smiles, isomer=True, num_enantiomers=1):
    """
    Generates an set of conformers from a SMILES string
    """
    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smiles):
        raise ValueError(f"SMILES invalid for string {smiles}")
    else:
        return from_mol(mol, isomer, num_enantiomers)


def from_structure(structure_file: Path):# -> oechem.OEMol:
    """
    Generates an set of conformers from a SMILES string
    """
    mol = oechem.OEMol()
    ifs = oechem.oemolistream()
    if not ifs.open(str(structure_file)):
        raise ValueError(f"Could not open structure file: {structure_file}")

    if structure_file.suffix == ".pdb":
        oechem.OEReadPDBFile(ifs, mol)
    elif structure_file.suffix == ".sdf":
        oechem.OEReadMDLFile(ifs, mol)
    else:
        raise ValueError(f"Invalid structure file extension: {structure_file}")

    return mol


def select_enantiomer(mol_list):
    return mol_list[0]


def dock_conf(receptor, mol, max_poses: int = 1):
    dock = oedocking.OEDock()
    dock.Initialize(receptor)
    lig = oechem.OEMol()
    err = dock.DockMultiConformerMolecule(lig, mol, max_poses)
    return dock, lig

# Returns an array of length max_poses from above. This is the range of scores
def ligand_scores(dock, lig):
    return [dock.ScoreLigand(conf) for conf in lig.GetConfs()]


def best_dock_score(dock, lig):
    return ligand_scores(dock, lig)#[0]


def write_ligand(ligand, output_dir: Path, smiles: str, lig_identify: str) -> None:
    # TODO: If MAX_POSES != 1, we should select the top pose to save
    ofs = oechem.oemolostream()
    for it, conf in enumerate(list(ligand.GetConfs())):
        if ofs.open(f'{str(output_dir)}/{lig_identify}/{it}.pdb'):
            oechem.OEWriteMolecule(ofs, conf)
            ofs.close()
    return
    raise ValueError(f"Could not write ligand to {output_path}")


def write_receptor(receptor, output_path: Path) -> None:
    ofs = oechem.oemolostream()
    if ofs.open(str(output_path)):
        mol = oechem.OEMol()
        contents = receptor.GetComponents(mol)#Within
        oechem.OEWriteMolecule(ofs, mol)
        ofs.close()
    return
    raise ValueError(f"Could not write receptor to {output_path}")

def run_test(smiles):
    return -1


@cache  # Only read the receptor once
def read_receptor(receptor_oedu_file: Path):
    """Read the .oedu file into a GraphMol object."""
    receptor = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(str(receptor_oedu_file), receptor)
    return receptor

#@cache
def create_proteinuniv(protein_pdb):
    protein_universe = mda.Universe(protein_pdb)
    return protein_universe

def create_complex(protein_universe, ligand_pdb):
    u1 = protein_universe
    u2 = mda.Universe(ligand_pdb)
    u = mda.core.universe.Merge(u1.select_atoms("all"), u2.atoms)#, u3.atoms)
    return u

def create_trajectory(protein_universe, ligand_dir, output_pdb_name, output_dcd_name):
    import MDAnalysis as mda
    ligand_files = sorted(os.listdir(ligand_dir))
    comb_univ_1 = create_complex(protein_universe, f'{ligand_dir}/{ligand_files[0]}').select_atoms("all")

    with mda.Writer(output_pdb_name, comb_univ_1.n_atoms) as w:
        w.write(comb_univ_1)
    with mda.Writer(output_dcd_name, comb_univ_1.n_atoms,) as w:
        for it, ligand_file in enumerate(ligand_files):
            comb_univ = create_complex(protein_universe, f'{ligand_dir}/{ligand_file}')
            w.write(comb_univ)    # write a whole universe
            os.remove(f'{ligand_dir}/{ligand_file}')
    return


def continue_simulations(stop_event, sim_iter, worker_logger=None):
    """Check if the continue event is set to continue simulations."""
    sequential_workflow = stop_event is None
    if worker_logger:
            worker_logger.info(f"Sequential workflow is {sequential_workflow}")
    if sequential_workflow:
        if sim_iter == 0:
            return True
        else:
            return False
    else:
        if worker_logger:
            worker_logger.info(f"Continue is {stop_event.is_set() is False}")
        return stop_event.is_set() is False 

def run_docking(sim_dd, 
                model_list_dd, 
                proc: int, 
                num_procs: int, 
                update_barrier=None, 
                stop_event=None,
                new_model_event=None,
                checkpoint_barrier=None,
                list_poll_interval_sec=10):
    
    """Run docking simulations using OpenEye.
    :param sim_dd: Dragon distributed dictionary for simulation results
    :type dd: DDict
    :param model_list_dd: Dragon distributed dictionary for model list and checkpoints
    :type model_list_dd: DDict
    :param proc: process group process integer
    :type proc: int
    :param num_procs: total number of docking processes
    :type num_procs: int
    :param update_barrier: multiprocessing barrier to synchronize updates to simulated compounds list
    :type update_barrier: mp.Barrier
    :param continue_event: multiprocessing event to signal whether to continue simulations
    :type continue_event: mp.Event
    :param new_model_event: multiprocessing event to signal new model is available
    :type new_model_event: mp.Event
    :param checkpoint_barrier: multiprocessing barrier to synchronize at checkpoints
    :type checkpoint_barrier: mp.Barrier
    :param list_poll_interval_sec: time in seconds to wait before polling for new top candidates
    :type list_poll_interval_sec: int
    """

    sequential_workflow = stop_event is None
    tic_start = perf_counter()

    # Setup logger for this worker
    os.makedirs("dock_worker_logs", exist_ok=True)
    worker_logger = setup_logger(f'dock_worker_{proc}', 
                                f"dock_worker_logs/dock_worker_{proc}.log", 
                                level=logging.DEBUG)
    myp = current_process()
    p = psutil.Process()
    core_list = p.cpu_affinity()
    hostname = socket.gethostname()
    worker_logger.info(f"Launching for worker {proc} from process {myp.ident} on core {core_list} on device {hostname}")
      
    # Start running simulations
    prev_top_candidates = []
    sim_iter = 0
    while continue_simulations(stop_event, sim_iter):
        worker_logger.info(f"{sim_iter=} Starting iteration...")
        checkpoint_id = model_list_dd.checkpoint_id
        worker_logger.info(f"{sim_iter=} On checkpoint {checkpoint_id}...")

        # Check for a new model and checkpoint model_list_dd if found
        if not sequential_workflow:
            if new_model_event.is_set():
                model_list_dd.checkpoint()
                checkpoint_id = model_list_dd.checkpoint_id
                worker_logger.info(f"{sim_iter=} Detected new model, updating to checkpoint {checkpoint_id}")
                pause_checkpointing = True

        # Get current top candidates
        # This will block until current_sort_list is avilable in the checkpoint
        top_candidates = model_list_dd.bget("current_sort_list")
        if top_candidates == prev_top_candidates:
            worker_logger.info(f"{sim_iter=} No new top candidates found, sleeping for {list_poll_interval_sec} seconds...")
            time.sleep(list_poll_interval_sec)
            #sim_iter += 1
            continue
        else:
            prev_top_candidates = top_candidates.copy()
            worker_logger.info(f"{sim_iter=} New top candidates found, proceeding with docking...")
        top_candidates_list = list(zip(top_candidates['smiles'], top_candidates['inf'], top_candidates['model_iter']))
        worker_logger.info(f"{sim_iter=} Sorted list has {len(top_candidates_list)} candidates")
        
        ckeys = list(model_list_dd.keys())
        # Add random samples to sorted candidates if available
        if "random_compound_sample" in ckeys:
            random_candidates = model_list_dd['random_compound_sample']
            random_candidates_list = list(zip(random_candidates['smiles'],random_candidates['inf'],random_candidates['model_iter']))
            worker_logger.info(f"{sim_iter=} Random candidate list has {len(random_candidates_list)} candidates")
            top_candidates_list += random_candidates_list
        else:
            worker_logger.info(f"{sim_iter=} No random candidate list found")                
        
        # Create top candidate dictionary for easy lookup
        top_candidates_dict = {}
        for i in range(len(top_candidates_list)):
            cand = top_candidates_list[i]
            top_candidates_dict[cand[0]] = (cand[1],cand[2])
        top_candidates_smiles = list(top_candidates_dict.keys())

        # Remove top candidates that have already been simulated
        worker_logger.info(f"{sim_iter=} Found {len(top_candidates_smiles)} top candidates")
            
        # Partition top candidate list to get candidates for this process to simulate
        num_candidates = len(top_candidates_smiles)
        if num_procs < num_candidates:
            my_candidates = split_dict_keys(top_candidates_smiles, num_procs, proc)
        else:
            if proc < len(top_candidates_smiles):
                my_candidates = [top_candidates_smiles[proc]]
            else:
                my_candidates = []
        worker_logger.debug(f"{sim_iter=} Assigned {len(my_candidates)} candidates")    
        
        # If there are assigned candidates to simulate, run sims
        if len(my_candidates) > 0:
            tic = perf_counter()
            if not os.getenv("DOCKING_SIM_DUMMY"):
                sim_metrics = dock(sim_dd, my_candidates, top_candidates_dict, proc, worker_logger) 
            else:
                sim_metrics = dummy_dock(sim_dd, my_candidates, top_candidates_dict, proc, worker_logger) 
            toc = perf_counter()
            worker_logger.debug(f"{sim_iter=} docking_sim_time {toc-tic} s")
        else:
            worker_logger.debug(f"{sim_iter=} no sims run \n") 

        worker_logger.info(f"{sim_iter=} Docking worker {proc} completed iteration \n")        
        sim_iter += 1

    toc_end = perf_counter()
    worker_logger.info(f"{toc_end-tic_start}")
    return



def dock(sdd: DDict, candidates: List[str], top_candidates_dict: dict, proc: int, worker_logger, debug=False):
    """Run OpenEye docking on a single ligand.

    Parameters
    ----------
    sdd : DDict
        A Dragon Dictionary to store results
    candidates : List[str]
        A list of smiles strings to be processed.
    sim_candidates : List[str]
        A list of smiles strings to be simulated.
    top_candidates_dict : dict
        A dictionary of top candidates with smiles as keys and inferred scores as values.
    proc : int
        Process group process integer
    worker_logger : logging.Logger
        Logger for this worker process.
    
    Returns
    -------
    dict
        A dictionary with metrics on performance
    """

    worker_logger.info(f"Docking worker {proc} starting docking of {len(candidates)} candidates...")
    num_cand = len(candidates)

    tic = perf_counter()
    receptor_oedu_file = os.getenv("RECEPTOR_FILE")

    max_confs = 1

    simulated_smiles = []
    dock_scores = []

    data_store_time = 0
    data_store_size = 0

    smiter = 0
    num_simulated = 0
    for smiles in candidates:
        dtic = perf_counter()
        try:
            val = sdd[smiles]
            dock_score = val['dock_score']
            inf_scores = val['inf_scores']
            if top_candidates_dict[smiles] not in inf_scores:
                inf_scores.append(top_candidates_dict[smiles])
            worker_logger.debug(f"{smiter+1}/{num_cand}: Candidate already simulated")
        except KeyError:
            num_simulated += 1
            with stdout_to_logger(worker_logger, level=logging.INFO):
                try:
                    try:
                        conformers = select_enantiomer(from_string(smiles))
                    except:
                        worker_logger.info(f"Conformers failed in batch {batch_key}, returning 0 docking score")
                        simulated_smiles.append(smiles)

                        dock_score = 0
                    
                        # Not implementing this alternate way of getting conformers for now
                        # with tempfile.NamedTemporaryFile(suffix=".pdb", dir=temp_storage) as fd:
                        #     # Read input SMILES and generate conformer
                        #     smi_to_structure(smiles, Path(fd.name))
                        #     conformers = from_structure(Path(fd.name))
                    else:
                        # Read the receptor to dock to
                        receptor = read_receptor(receptor_oedu_file)
                        # Dock the ligand conformers to the receptor
                        dock, lig = dock_conf(receptor, conformers, max_poses=max_confs)

                        # Get the docking scores
                        best_score = best_dock_score(dock, lig)

                        simulated_smiles.append(smiles)
                        dock_score = max(-1*np.mean(best_score),0.)
                        
                except:
                    simulated_smiles.append(smiles)
                    dock_score = 0
            dock_scores.append(dock_score)
            inf_scores = [top_candidates_dict[smiles]]
            dtoc = perf_counter()
            worker_logger.debug(f"{smiter+1}/{num_cand}: Docking performed in {dtoc-dtic} seconds")
            
        sdd[smiles] = {'dock_score':dock_score, 'inf_scores':inf_scores}
        #worker_logger.debug(f"{smiles=}, {dock_score=}")
        dtoc = perf_counter()
        ddict_time = dtoc-dtic
        ddict_size = sys.getsizeof(smiles) + sys.getsizeof(dock_score)
        smiter += 1
        
    toc = perf_counter()
    time_per_cand = (toc-tic)/num_cand
    worker_logger.debug(f"All candidates completed: {smiter == num_cand}")

    new_keys = sdd.keys()
    num_sim = 0
    for smiles in candidates:
        if smiles in new_keys:
            num_sim += 1
    worker_logger.debug(f"Candidates in keys: {num_sim}/{num_cand}")
    metrics = {}
    metrics['total_run_time'] = toc-tic
    metrics['num_cand'] = num_cand
    metrics['ddict_time'] = ddict_time
    metrics['dict_size'] =  ddict_size

    worker_logger.debug(f"{dock_scores=}")
    worker_logger.debug(f"Simulated {num_simulated} out of {num_cand} candidates in {toc-tic} s, {time_per_cand=}\n")

    #worker_logger.info(f"Simulated {num_cand} candidates in {toc-tic} s on worker {proc}, {time_per_cand=}")
    return metrics



def dummy_dock(sdd, candidates, top_candidates_dict, proc: int, worker_logger, debug=False):
    """Mock docking function that simulates docking time with sleep and generates dummy scores.
    
    Parameters
    ----------
    sdd : DDict
        A Dragon Dictionary to store results
    candidates : List[str]
        A list of smiles strings to be processed.
    sim_candidates : List[str]
        A list of smiles strings to be simulated.
    top_candidates_dict : dict
        A dictionary of top candidates with smiles as keys and inferred scores as values.
    proc : int
        Process group process integer
    worker_logger : logging.Logger
        Logger for this worker process.
    
    Returns
    -------
    dict
        A dictionary with metrics on performance
  
    """

    worker_logger.info(f"Generating dummy dock scores with sleep")
    num_cand = len(candidates)

    tic = perf_counter()

    simulated_smiles = []
    dock_scores = []

    data_store_time = 0
    data_store_size = 0

    smiter = 0
    for smiles in candidates:
        #worker_logger.debug(f"Processing candidate {smiter+1}/{num_cand}: {smiles}")
        dtic = perf_counter()
        try:
            val = sdd[smiles]
            dock_score = val['dock_score']
            inf_scores = val['inf_scores']
            inf_scores.append(top_candidates_dict[smiles])
        except KeyError:
            # We will add a random offset to inferred docking score
            dock_score_offset = random.uniform(-0.5, 0.5)
            dock_score = top_candidates_dict[smiles][0] + dock_score_offset
            
            # Sleep for 6-8 seconds to simulate docking time
            sleep_offset = random.uniform(-1., 1.)
            time.sleep(7.+sleep_offset)
            inf_scores = [top_candidates_dict[smiles]]
            dtoc = perf_counter()
            worker_logger.debug(f"{smiter+1}/{num_cand}: Docking performed in {dtoc-dtic} seconds")
           
        #worker_logger.debug(f"Storing result for candidate {smiter+1}/{num_cand}: {smiles} with dock score {dock_score}")
        sdd[smiles] = {'dock_score':dock_score, 'inf_scores':inf_scores}
        #worker_logger.debug(f"Result stored for candidate {smiter+1}/{num_cand}: {smiles}")
        #worker_logger.debug(f"{smiles=} {dock_score=}")
        #data_store_time += dtic-dtoc
        #data_store_size += sys.getsizeof(smiles) + sys.getsizeof(dock_score)
        
        smiter += 1

    toc = perf_counter()
    time_per_cand = (toc-tic)

    metrics = {}
    #metrics['total_run_time'] = toc-tic
    #metrics['num_cand'] = num_cand
    #metrics['data_store_time'] = data_store_time
    #metrics['data_store_size'] =  data_store_size

    worker_logger.info(f"Processed {num_cand} candidates in {toc-tic} s")

    return metrics

if __name__ == "__main__":

    import pathlib
    import gzip
    import glob
    
    num_procs = 1
    proc = 0
    continue_event = None
    dd = {}

    file_dir = os.getenv("DATA_PATH")
    all_files = glob.glob(file_dir+"*.gz")
    file_path = all_files[0]
        
    smiles = []
    f_name = str(file_path).split("/")[-1]
    f_extension = str(file_path).split("/")[-1].split(".")[-1]
    if f_extension=="smi":
        with file_path.open() as f:
            for line in f:
                smile = line.split("\t")[0]
                smiles.append(smile)
    elif f_extension=="gz":
        with gzip.open(str(file_path), 'rt') as f:
            for line in f:
                smile = line.split("\t")[0]
                smiles.append(smile)

    candidates = smiles[0:10]
    cdd = {'asldkfjas;ld_fake_smiles': 11.2,
           smiles[0]: 10.7, #fake dock score
           smiles[1]: 8.5,
           '0': candidates,
           'cmax_iter': 0,
          }
    docking_iter = 0
    proc = 0
    num_procs = 1
    
    run_docking(cdd, docking_iter, proc, num_procs)
