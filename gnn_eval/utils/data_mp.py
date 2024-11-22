import os
import pandas as pd
from tqdm import tqdm
import pickle as pkl
try:
    from mp_api.client import MPRester
except ImportError:
    print("Warning: MPRester not imported. Ensure the `mp_api` library is installed if using Materials Project API.")
from utils.data import pm2ase

# Initialize the MPRester
def mp_struct_prop(api_key, mat_input=None, use_ase=True, save_file=None):
    with MPRester(api_key) as mpr:
        mp_dict = {}

        # Determine whether to use a list of material IDs or fetch a number of materials based on `material_input`
        if isinstance(mat_input, list):  # List of material IDs provided
            materials_to_fetch = mat_input
            fetch_by_id = True
        elif isinstance(mat_input, int):  # Maximum number of materials provided
            materials_to_fetch = range(mat_input)  # Use range to generate indices
            fetch_by_id = False
            summaries = mpr.summary.search(fields=["material_id", "energy_above_hull", "formation_energy_per_atom", 'band_gap', 'ordering'], all_fields=False)
        else:
            raise ValueError("mat_input must be either a list of material IDs or an integer representing the number of materials to fetch.")
        for i, identifier in tqdm(enumerate(materials_to_fetch), desc="Processing"):
            try:
                if fetch_by_id:
                    # Fetching by material ID
                    mpid = identifier
                    pstruct = mpr.get_structure_by_material_id(mpid)
                    summary = mpr.summary.get_data_by_id(mpid, fields=["material_id", "energy_above_hull", "formation_energy_per_atom", 'band_gap', 'ordering'])
                else:
                    # Fetching by index, only for demo purposes; replace with actual logic to fetch materials without IDs
                    # This branch assumes you're fetching the first `material_input` materials, adjust according to your API's capabilities
                    # summary = mpr.summary.search(fields=["material_id", "energy_above_hull", "formation_energy_per_atom"], all_fields=False)[i]
                    summary = summaries[i]
                    mpid = summary.material_id
                    pstruct = mpr.get_structure_by_material_id(mpid)

                e_above_hull = summary.energy_above_hull
                f_energy_per_atom = summary.formation_energy_per_atom
                egap = summary.band_gap
                mag = summary.ordering if summary.ordering else 'Unknown'

                if pd.isna(e_above_hull) or pd.isna(f_energy_per_atom):
                    continue

                if use_ase:
                    struct = pm2ase(pstruct)
                else:
                    struct = pstruct

                mp_dict[mpid] = {"structure": struct, "ehull": e_above_hull, 'f_energy': f_energy_per_atom, 'egap': egap, 'mag': mag}
                print(f'[{i}] mpid:', mpid, 'ehull:', e_above_hull, 'f_energy: ', f_energy_per_atom, 'egap:', egap, 'mag:', mag)
            except Exception as e:
                print(f'Error loading material with ID {mpid}: {e}')

    data_for_df = []
    for mpid, values in mp_dict.items():
        # Assuming you want to keep the full mpid string, otherwise extract the numeric part
        # For the structure, you might want to use something like values['structure'].formula if it's an actual object
        row = {
            "mpid": mpid,
            "structure": values["structure"],  # This is simplified for the example
            "ehull": values["ehull"],
            "f_energy": values["f_energy"],
            "egap": values["egap"],
            "mag": values["mag"]
        }
        data_for_df.append(row)
    # Step 3: Convert to DataFrame
    mpdata = pd.DataFrame(data_for_df)
    
    if isinstance(save_file, str):
        with open(save_file, 'wb') as file:
            pkl.dump(mpdata, file)

    return mpdata


def mp_struct_prop_stable_filter(api_key, use_ase=True, save_file=None, s_u_nums=[2000, 1000], natm_max=20):
    num_stable, num_unstable = s_u_nums
    with MPRester(api_key) as mpr:
        mp_dict = {}
        stable_materials = []
        unstable_materials = []
        summaries =  mpr.summary.search(fields=["material_id", "energy_above_hull", "formation_energy_per_atom", "nsites", 'band_gap', 'ordering'], all_fields=False)

        max_attempts = 10000  # Set a maximum number of attempts to prevent infinite loops
        attempts = 0
        while len(stable_materials) < num_stable or len(unstable_materials) < num_unstable:
            if attempts >= max_attempts:
                print("Reached maximum number of attempts")
                break
            try:
                # Adjust fetching logic here. This is a placeholder for fetching a single material at a time
                # You might want to implement a more efficient batch-fetching and filtering strategy
                identifier = attempts  # Placeholder for actual material ID or index
                # Fetch material data by ID or another identifier
                # summary = mpr.summary.search(fields=["material_id", "energy_above_hull", "formation_energy_per_atom", "nsites"], all_fields=False)[identifier]
                summary = summaries[identifier]
                mpid = summary.material_id
                e_above_hull = summary.energy_above_hull
                nsites = summary.nsites  # Number of atoms per unit cell

                if pd.isna(e_above_hull) or pd.isna(nsites):
                    continue

                # Filtering based on stability and number of atoms
                if nsites < natm_max:
                    if e_above_hull < 0.1 and len(stable_materials) < num_stable:
                        stable_materials.append(mpid)
                        print(f'[{attempts}] stable :', mpid, 'ehull:', e_above_hull)
                    elif e_above_hull >= 0.1 and len(unstable_materials) < num_unstable:
                        unstable_materials.append(mpid)
                        print(f'[{attempts}] unstable :', mpid, 'ehull:', e_above_hull)
                else: 
                    print(f'XX [{attempts}] not applicable :', mpid, 'ehull:', e_above_hull)   

            except Exception as e:
                print(f'Error loading material with ID {mpid}: {e}')
            
            attempts += 1

        # Now fetch the detailed data for each selected material
        for i, mpid in tqdm(enumerate(stable_materials + unstable_materials), desc="Fetching detailed data"):
            try:
                pstruct = mpr.get_structure_by_material_id(mpid)
                summary = mpr.get_data_by_id(mpid, fields=["material_id", "energy_above_hull", "formation_energy_per_atom"])
                e_above_hull = summary.energy_above_hull
                f_energy_per_atom = summary.formation_energy_per_atom
                egap = summary.band_gap
                mag = summary.ordering if summary.ordering else 'Unknown'

                if use_ase:
                    struct = pm2ase(pstruct)
                else:
                    struct = pstruct

                mp_dict[mpid] = {"structure": struct, "ehull": e_above_hull, 'f_energy': f_energy_per_atom, 'egap': egap, 'mag': mag}
                print(f'[{i}] mpid:', mpid, 'ehull:', e_above_hull, 'f_energy: ', f_energy_per_atom, 'egap:', egap, 'mag:', mag)
            except Exception as e:
                print(f'Error loading material with ID {mpid}: {e}')

        data_for_df = []
        for mpid, values in mp_dict.items():
            row = {
                "mpid": mpid,
                "structure": values["structure"],
                "ehull": values["ehull"],
                "f_energy": values["f_energy"],
                "egap": values["egap"],
                "mag": values["mag"]
            }
            data_for_df.append(row)

        mpdata = pd.DataFrame(data_for_df)
    
    if save_file:
        with open(save_file, 'wb') as file:
            pkl.dump(mpdata, file)

    return mpdata