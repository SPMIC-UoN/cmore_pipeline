"""
Processing pipeline for CMORE data
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import traceback

import numpy as np
import nibabel as nib

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="demistifi-ukb-pipeline", add_help=True, **kwargs)
        self.add_argument("--input", required=True, help="Input directory containing subject dirs")
        self.add_argument("--output", required=True, help="Output directory")
        self.add_argument("--subjids", help="File containing subject IDs to process. If not specified process all subjects")
        self.add_argument("--subjid-idx", type=int, help="Index of individual subject ID to process (starting at 1). If not specified, process all")        
        self.add_argument("--skip-preproc", action='store_true', default=False, help="Skip renal-preproc step")
        self.add_argument("--skip-seg", action='store_true', default=False, help="Skip T1 segmentation steps")
        self.add_argument("--skip-stats", action='store_true', default=False, help="Skip statistics generation")
        self.add_argument("--seg-models-dir", default="/spmstore/project/RenalMRI/cmore_trained_models", help="Directory contained trained segmentation models")

def link(srcdir, srcfile, destdir, destfile, multiple_ok=False):
    """
    Link supporting wildcards
    """
    srcfiles = list(glob.glob(os.path.join(srcdir, f"{srcfile}.nii.gz")))
    if not srcfiles:
        print(f"WARNING: Could not create link for output file {destfile} - source file {srcfile} not found")
    elif len(srcfiles) > 1 and not multiple_ok:
        print(f"WARNING: Could not create link for output file {destfile} - multiple source files {srcfile} found")
    else:
        os.symlink(os.path.abspath(srcfiles[0]), os.path.join(destdir, f"{destfile}.nii.gz"))

def handle_r2star_t2star(indir):
    """
    Make sure R2* units are in s^-1 and calculate T2* from R2* if missing
    """
    for root, dirs, files in os.walk(indir):
        for f in files:
            if "r2star" in f:
                f_r2star = os.path.join(root, f)
                try:
                    nii_r2star = nib.load(f_r2star)
                    data_r2star = nii_r2star.get_fdata()
                    med = np.median(data_r2star)
                    if med < 1:
                        print(f"R2* data median {med} - assuming ms^-1, converting to s^-1")
                        nii_r2star = nib.Nifti1Image(1000.0 * data_r2star, None, nii_r2star.header)
                        nii_r2star.to_filename(f_r2star)
                except:
                    print(f"WARNING: Failed to correct R2* units for file: {f_r2star}")
                    traceback.print_exc()

                f_t2star = f_r2star.replace("r2star", "t2star")
                if not os.path.exists(f_t2star):
                    print(f"T2* not found for R2* file {f_r2star} - creating")
                    try:
                        nii_t2star = nib.Nifti1Image(1000.0/nii_r2star.get_fdata(), None, nii_r2star.header)
                        nii_t2star.to_filename(f_t2star)
                    except:
                        print(f"WARNING: Failed to calculate T2* for R2* file: {f_r2star}")
                        traceback.print_exc()

def run(cmd, logfile):
    """
    Run a command and raise exception if it fails
    """
    with open(logfile, "w") as f:
        retval = subprocess.call(cmd, stdout=f, stderr=f)
    if retval != 0:
        print(f"WARNING: command\n{cmd}\nreturned non-zero exit state {retval}")

def main():
    options = ArgumentParser().parse_args()

    if not options.subjids:
        subjids = []
        for site in os.listdir(options.input):
            sitedir = os.path.join(options.input, site)
            subjids += [d for d in os.listdir(sitedir) if os.path.isdir(os.path.join(sitedir, d))]
        subjids = sorted(subjids)
    else:
        with open(options.subjids, "r") as f:
            subjids = [l.strip() for l in f.readlines()]
    
    if options.subjid_idx:
        subjids = [subjids[options.subjid_idx-1]]

    for subjid in subjids:
        found = False
        for site in os.listdir(options.input):
            subjdir = os.path.join(options.input, site, subjid)
            if os.path.isdir(subjdir):
                found = True
                break

        if not found:
            print(f"WARNING: {subjid} not found in any site dir - skipping")
            continue

        print(f"Running subject {subjid} from site {site}")
        outdir = os.path.join(options.output, subjid)
        os.makedirs(outdir, exist_ok=True)

        if not options.skip_preproc:
            print(f"Doing renal preprocessing for subject {subjid}")
            model = os.path.join(options.seg_models_dir, "t2star_seg.h5")
            run(['renal-preproc',
                 '--indir', subjdir,
                 '--outdir', f'{outdir}/renal_preproc',
                 '--single-session',
                 '--segmentation-weights', model,
                 '--overwrite'], logfile=f'{outdir}/renal_logfile.txt')
            handle_r2star_t2star(outdir)
            print(f"DONE renal preprocessing for subject {subjid}")

        if not options.skip_seg:
            print(f"Doing kidney T1 segmentation for subject {subjid}")
            model = os.path.join(options.seg_models_dir, "t1_seg.pt")
            run(['kidney_t1_seg',
                 '--input', options.output,
                 '--subjid', subjid,
                 '--t1', 'renal_preproc/t1_out/*_t1map.nii*',
                 '--model', model,
                 '--output', options.output,
                 '--outprefix', 't1_seg/seg_kidney'], logfile=f'{outdir}/t1_seg_logfile.txt')
            print(f"DONE kidney T1 segmentation for subject {subjid}")

        if not options.skip_stats:
            print(f"Linking segmentation and data sets for subject {subjid}")
            qp_data_dir = os.path.join(outdir, "qpdata")
            if os.path.exists(qp_data_dir):
                shutil.rmtree(qp_data_dir)
            os.makedirs(qp_data_dir)
            renal_outdir = os.path.join(outdir, "renal_preproc")
            seg_outdir = os.path.join(outdir, "t1_seg")

            # Segmentations
            link(seg_outdir, f"seg_kidney_medulla_l_t1", qp_data_dir, "seg_kidney_medulla_l_t1")
            link(seg_outdir, f"seg_kidney_cortex_l_t1", qp_data_dir, "seg_kidney_cortex_l_t1")
            link(seg_outdir, f"seg_kidney_medulla_r_t1", qp_data_dir, "seg_kidney_medulla_r_t1")
            link(seg_outdir, f"seg_kidney_cortex_r_t1", qp_data_dir, "seg_kidney_cortex_r_t1")

            # Renal preproc outputs
            link(renal_outdir, "t2star_out/*_loglin_t2star_map", qp_data_dir, "t2star_loglin")
            link(renal_outdir, "t2star_out/*_exp_t2star_map", qp_data_dir, "t2star_exp")
            link(renal_outdir, "t2star_out/*_loglin_r2star_map", qp_data_dir, "r2star_loglin")
            link(renal_outdir, "t2star_out/*_exp_r2star_map", qp_data_dir, "r2star_exp")
            link(renal_outdir, "tkv_out/*_right_kidney", qp_data_dir, "seg_kidney_r_t2w")
            link(renal_outdir, "tkv_out/*_left_kidney", qp_data_dir, "seg_kidney_l_t2w")
            link(renal_outdir, "tkv_out/*_mask", qp_data_dir, "seg_kidney_t2w")
            link(renal_outdir, "t1_out/*_t1map", qp_data_dir, "t1", multiple_ok=True)

            print(f"DONE Linking segmentation and data sets for subject {subjid}")

            print(f"Extracting ROI stats for subject {subjid}")
            qp_script = os.path.join(os.path.dirname(sys.argv[0]), "resample_and_stats.qp")
            subj_qp_script = os.path.join(qp_data_dir, "resample_and_stats.qp")
            if os.path.exists(subj_qp_script):
                os.remove(subj_qp_script, exist_ok=True)
            with open(qp_script, "r") as f:
                with open(subj_qp_script, "w") as of:
                    for line in f.readlines():
                        of.write(line.replace("SUBJID", subjid).replace("OUTDIR", options.output))
            run(['quantiphyse', '--batch', f'{qp_data_dir}/resample_and_stats.qp'], logfile=f'{outdir}/qp_logfile.txt')
            print(f"DONE Extracting ROI stats for subject {subjid}")

        print(f"DONE running subject {subjid}")

if __name__ == "__main__":
    main()
