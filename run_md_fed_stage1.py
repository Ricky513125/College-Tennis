#!/usr/bin/env python3
"""
Helper script to run MD-FED Stage 1 training with data in current directory.
This script directly uses data from current directory without creating symbolic links.
"""

import os
import sys
import argparse


def check_data_files(data_dir, dataset_name='f3set-tennis-sub'):
    """Check if required data files exist"""
    dataset_dir = os.path.join(data_dir, dataset_name)
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Data directory not found: {dataset_dir}")
        print("Please run prepare_md_fed_data.py first")
        return False
    
    required_files = ['train.json', 'val.json', 'elements.txt']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(dataset_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files in {dataset_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print(f"✓ Data files found in {dataset_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run MD-FED Stage 1 training with data in current directory'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='md_fed_data',
        help='Directory containing prepared data (default: md_fed_data)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='f3set-tennis-sub',
        help='Dataset name (default: f3set-tennis-sub)'
    )
    parser.add_argument(
        '--pose_dir',
        type=str,
        required=True,
        help='Path to skeleton pkl files directory'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        default='frames',
        help='Path to RGB frames directory (default: frames)'
    )
    parser.add_argument(
        '--flow_dir',
        type=str,
        default='flows',
        help='Path to optical flow directory (default: flows)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='md_fed_outputs/stage1',
        help='Output directory for checkpoints and logs (default: md_fed_outputs/stage1)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--visual_arch',
        type=str,
        default='rny002_tsm',
        help='Visual architecture (default: rny002_tsm)'
    )
    parser.add_argument(
        '--skeleton_arch',
        type=str,
        default='stgcn++',
        help='Skeleton architecture (default: stgcn++)'
    )
    parser.add_argument(
        '--setup_only',
        action='store_true',
        help='Only check data files, do not run training'
    )
    
    args = parser.parse_args()
    
    # Check data files
    print("Checking data files...")
    data_dir = os.path.abspath(args.data_dir)
    if not check_data_files(data_dir, args.dataset):
        sys.exit(1)
    
    if args.setup_only:
        print("Data files check complete.")
        return
    
    # Change to MD-FED directory and import
    original_cwd = os.getcwd()
    md_fed_dir = os.path.join(original_cwd, 'MD-FED')
    if not os.path.exists(md_fed_dir):
        print(f"Error: MD-FED directory not found: {md_fed_dir}")
        sys.exit(1)
    
    os.chdir(md_fed_dir)
    sys.path.insert(0, md_fed_dir)
    
    try:
        # Import training modules
        import train_MD_FED
        from util.dataset import load_classes
        from dataset.input_process import ActionSeqDataset, ActionSeqVideoDataset
        
        # Patch get_datasets to use custom data directory
        original_get_datasets = train_MD_FED.get_datasets
        
        def patched_get_datasets(args):
            """Patched version that uses data from current directory"""
            elements_file = os.path.join(data_dir, args.dataset, 'elements.txt')
            train_json = os.path.join(data_dir, args.dataset, 'train.json')
            val_json = os.path.join(data_dir, args.dataset, 'val.json')
            
            classes = load_classes(elements_file)
            
            if 'f3set-tennis-sub' in args.dataset:
                epoch_num_frames = 500000 if (args.stage == 2 or args.num_samples == -1) else 50000
            elif 'shuttleset' in args.dataset:
                epoch_num_frames = 200000 if (args.stage == 2 or args.num_samples == -1) else 100000
            else:
                epoch_num_frames = 500000 if (args.stage == 2 or args.num_samples == -1) else 100000
            
            dataset_len = epoch_num_frames // (args.clip_len * args.stride)
            dataset_kwargs = {'crop_dim': args.crop_dim, 'stride': args.stride}
            
            print('Dataset size:', dataset_len)
            num_train_samples, num_val_samples = -1, -1
            if args.num_samples >= 0:
                num_train_samples = int(args.num_samples * 0.8)
                num_val_samples = args.num_samples - num_train_samples
            
            train_data = ActionSeqDataset(
                classes, train_json,
                args.frame_dir, args.clip_len, dataset_len, is_eval=False, 
                dilate_len=args.dilate_len, stage=args.stage,
                num_samples=num_train_samples, flow_dir=args.flow_dir, pose_dir=args.pose_dir,
                **dataset_kwargs)
            train_data.print_info()
            
            val_data = ActionSeqDataset(
                classes, val_json,
                args.frame_dir, args.clip_len, dataset_len // 4, 
                dilate_len=args.dilate_len, stage=args.stage, 
                num_samples=num_val_samples, flow_dir=args.flow_dir, pose_dir=args.pose_dir,
                **dataset_kwargs)
            val_data.print_info()
            
            val_data_frames = None
            if args.criterion == 'edit':
                val_data_frames = ActionSeqVideoDataset(
                    classes, val_json,
                    args.frame_dir, args.clip_len, overlap_len=0, num_samples=num_val_samples,
                    flow_dir=args.flow_dir, pose_dir=args.pose_dir, **dataset_kwargs)
            
            return classes, train_data, val_data, None, val_data_frames
        
        # Replace the function
        train_MD_FED.get_datasets = patched_get_datasets
        
        # Parse arguments for training
        import argparse as ap
        parser = ap.ArgumentParser()
        parser.add_argument('dataset', type=str)
        parser.add_argument('--frame_dir', type=str, default=args.frame_dir)
        parser.add_argument('--flow_dir', type=str, default=args.flow_dir)
        parser.add_argument('--pose_dir', type=str, default=os.path.abspath(args.pose_dir))
        parser.add_argument('--stage', type=int, default=1)
        parser.add_argument('--visual_arch', type=str, default=args.visual_arch)
        parser.add_argument('--skeleton_arch', type=str, default=args.skeleton_arch)
        parser.add_argument('--num_epochs', type=int, default=args.num_epochs)
        parser.add_argument('--batch_size', type=int, default=args.batch_size)
        parser.add_argument('--learning_rate', type=float, default=args.learning_rate)
        parser.add_argument('-s', '--save_dir', type=str, default=os.path.abspath(args.output_dir))
        parser.add_argument('--num_samples', type=int, default=-1)
        parser.add_argument('--clip_len', type=int, default=96)
        parser.add_argument('--crop_dim', type=int, default=224)
        parser.add_argument('--stride', type=int, default=2)
        parser.add_argument('--dilate_len', type=int, default=0)
        parser.add_argument('--criterion', type=str, default='edit')
        parser.add_argument('--window', type=int, default=5)
        parser.add_argument('--temporal_arch', type=str, default='gru')
        parser.add_argument('--acc_grad_iter', type=int, default=1)
        parser.add_argument('--warm_up_epochs', type=int, default=3)
        parser.add_argument('--start_val_epoch', type=int, default=None)
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--gpu_parallel', action='store_true', default=False)
        parser.add_argument('--num_workers', type=int, default=None)
        
        # Create args object
        train_args = parser.parse_args([
            args.dataset,
            '--frame_dir', args.frame_dir,
            '--flow_dir', args.flow_dir,
            '--pose_dir', os.path.abspath(args.pose_dir),
            '--stage', '1',
            '--visual_arch', args.visual_arch,
            '--skeleton_arch', args.skeleton_arch,
            '--num_epochs', str(args.num_epochs),
            '--batch_size', str(args.batch_size),
            '--learning_rate', str(args.learning_rate),
            '-s', os.path.abspath(args.output_dir)
        ])
        
        print("\n" + "="*60)
        print("Starting MD-FED Stage 1 training...")
        print("="*60)
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {os.path.abspath(args.output_dir)}")
        print("="*60 + "\n")
        
        # Run training
        train_MD_FED.main(train_args)
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    main()
