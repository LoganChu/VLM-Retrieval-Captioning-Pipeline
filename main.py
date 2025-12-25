"""
Main pipeline orchestration script
Runs complete end-to-end pipeline with all components

Total time budget: 6-8 hours
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
import numpy as np

# Import all modules
from config import Config
from data_preparation import prepare_datasets, create_dataloaders
from clip_baseline import run_clip_baseline, CLIPRetrieval
from linear_probe import run_linear_probe, LinearProbeTrainer
from captioner import run_captioner
from evaluation import (
    evaluate_with_calibration,
    aggregate_results_across_seeds,
    compute_cider_score
)
from profiling import (
    GPUMonitor,
    profile_section,
    benchmark_inference,
    create_performance_report
)

class PipelineRunner:
    """Main pipeline runner"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {
            'clip_baseline': [],
            'linear_probe': [],
            'captioner_full': [],
            'captioner_lora': [],
        }
        self.gpu_monitor = GPUMonitor()
        self.training_times = {}
        
        # Create directories
        for dir_path in [config.experiment.output_dir,
                        config.experiment.log_dir,
                        config.experiment.checkpoint_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def run_data_preparation(self):
        """Step 1: Prepare datasets (30-45 min)"""
        print("\n" + "="*80)
        print("STEP 1: DATA PREPARATION")
        print("="*80)
        
        with profile_section("Data Preparation", self.gpu_monitor):
            train_data, val_data, test_data = prepare_datasets(self.config)
            
            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(
                train_data, val_data, test_data, self.config
            )
        
        return train_loader, val_loader, test_loader, train_data, val_data, test_data
    
    def run_clip_baseline_all_seeds(self, train_loader, val_loader, test_loader):
        """Step 2: CLIP zero-shot baseline with 3 seeds (45-60 min)"""
        print("\n" + "="*80)
        print("STEP 2: CLIP ZERO-SHOT BASELINE")
        print("="*80)
        
        start_time = time.time()
        
        for seed in self.config.experiment.seeds:
            with profile_section(f"CLIP Baseline - Seed {seed}", self.gpu_monitor):
                # First, encode training data and save
                clip_retrieval = CLIPRetrieval(self.config)
                
                # Encode training data
                train_image_embs, _ = clip_retrieval.encode_images(train_loader, "train")
                train_captions = [batch['caption'] for batch in train_loader.dataset]
                train_text_embs = clip_retrieval.encode_texts(train_captions)
                
                # Save training embeddings
                train_embs_path = os.path.join(
                    self.config.experiment.output_dir,
                    f'train_embeddings_seed{seed}.pt'
                )
                torch.save({
                    'image': train_image_embs,
                    'text': train_text_embs
                }, train_embs_path)
                
                # Run evaluation
                results = run_clip_baseline(
                    self.config, train_loader, val_loader, test_loader, seed
                )
                self.results['clip_baseline'].append(results)
        
        self.training_times['clip_baseline'] = time.time() - start_time
        
        # Aggregate results
        val_results = [r['val_metrics'] for r in self.results['clip_baseline']]
        test_results = [r['test_metrics'] for r in self.results['clip_baseline']]
        
        # Print aggregate statistics
        print("\n" + "="*60)
        print("CLIP Baseline - Aggregated Results")
        print("="*60)
        
        for metric in ['recall@1', 'recall@5', 'recall@10']:
            val_values = [r[metric] for r in val_results]
            test_values = [r[metric] for r in test_results]
            
            print(f"\n{metric}:")
            print(f"  Val:  {np.mean(val_values):.2f} ± {np.std(val_values):.2f}%")
            print(f"  Test: {np.mean(test_values):.2f} ± {np.std(test_values):.2f}%")
    
    def run_linear_probe_all_seeds(self):
        """Step 3: Linear probe with 3 seeds (1-1.25 hr)"""
        print("\n" + "="*80)
        print("STEP 3: LINEAR PROBE TRAINING")
        print("="*80)
        
        start_time = time.time()
        
        for seed in self.config.experiment.seeds:
            with profile_section(f"Linear Probe - Seed {seed}", self.gpu_monitor):
                results = run_linear_probe(self.config, seed)
                if results:
                    self.results['linear_probe'].append(results)
        
        self.training_times['linear_probe'] = time.time() - start_time
        
        # Aggregate results
        if self.results['linear_probe']:
            test_results = [r['test_metrics'] for r in self.results['linear_probe']]
            
            print("\n" + "="*60)
            print("Linear Probe - Aggregated Results")
            print("="*60)
            
            for metric in ['recall@1', 'recall@5', 'recall@10']:
                values = [r[metric] for r in test_results]
                print(f"{metric}: {np.mean(values):.2f} ± {np.std(values):.2f}%")
    
    def run_captioner_all_seeds(self, train_data, val_data, test_data):
        """Step 4: Captioner training with full-ft and LoRA (1.5-2 hr)"""
        print("\n" + "="*80)
        print("STEP 4: IMAGE CAPTIONING TRAINING")
        print("="*80)
        
        # Run full fine-tuning
        print("\n" + "-"*60)
        print("Full Fine-tuning")
        print("-"*60)
        
        start_time = time.time()
        for seed in self.config.experiment.seeds:
            with profile_section(f"Captioner Full-FT - Seed {seed}", self.gpu_monitor):
                results = run_captioner(
                    self.config, train_data, val_data, test_data, 
                    seed, use_lora=False
                )
                self.results['captioner_full'].append(results)
        self.training_times['captioner_full'] = time.time() - start_time
        
        # Run LoRA fine-tuning
        print("\n" + "-"*60)
        print("LoRA Fine-tuning")
        print("-"*60)
        
        start_time = time.time()
        for seed in self.config.experiment.seeds:
            with profile_section(f"Captioner LoRA - Seed {seed}", self.gpu_monitor):
                results = run_captioner(
                    self.config, train_data, val_data, test_data,
                    seed, use_lora=True
                )
                self.results['captioner_lora'].append(results)
        self.training_times['captioner_lora'] = time.time() - start_time
        
        # Compare results
        self.compare_captioner_results()
    
    def compare_captioner_results(self):
        """Compare full fine-tuning vs LoRA"""
        print("\n" + "="*60)
        print("Captioner Comparison: Full-FT vs LoRA")
        print("="*60)
        
        # Extract CIDEr scores
        full_ciders = []
        lora_ciders = []
        
        for full_result in self.results['captioner_full']:
            gen_captions = full_result['test_metrics']['generated_captions']
            ref_captions = full_result['test_metrics']['reference_captions']
            cider = compute_cider_score(gen_captions, ref_captions)
            full_ciders.append(cider)
        
        for lora_result in self.results['captioner_lora']:
            gen_captions = lora_result['test_metrics']['generated_captions']
            ref_captions = lora_result['test_metrics']['reference_captions']
            cider = compute_cider_score(gen_captions, ref_captions)
            lora_ciders.append(cider)
        
        print(f"\nCIDEr Score:")
        print(f"  Full-FT: {np.mean(full_ciders):.2f} ± {np.std(full_ciders):.2f}")
        print(f"  LoRA:    {np.mean(lora_ciders):.2f} ± {np.std(lora_ciders):.2f}")
        
        # Compare training times
        print(f"\nTraining Time:")
        print(f"  Full-FT: {self.training_times['captioner_full']:.2f}s ({self.training_times['captioner_full']/60:.2f} min)")
        print(f"  LoRA:    {self.training_times['captioner_lora']:.2f}s ({self.training_times['captioner_lora']/60:.2f} min)")
    
    def run_calibration_and_ood_analysis(self, test_loader):
        """Step 5: Calibration and OOD evaluation (30-45 min)"""
        print("\n" + "="*80)
        print("STEP 5: CALIBRATION & OOD ANALYSIS")
        print("="*80)
        
        with profile_section("Calibration Analysis", self.gpu_monitor):
            # Load best CLIP baseline results
            seed = self.config.experiment.seeds[0]
            embeddings_path = os.path.join(
                self.config.experiment.output_dir,
                f'clip_embeddings_seed{seed}.pt'
            )
            
            embs = torch.load(embeddings_path)
            
            # Prepare outputs for evaluation
            clip_retrieval = CLIPRetrieval(self.config)
            similarity = clip_retrieval.compute_similarity(
                embs['test_image_emb'],
                embs['test_text_emb']
            )
            
            model_outputs = {
                'similarity': similarity,
                'captions': embs['test_captions'],
                'is_ood': embs['test_is_ood']
            }
            
            # Run comprehensive evaluation
            eval_results = evaluate_with_calibration(
                model_outputs,
                self.config,
                split_name="test"
            )
            
            # Save evaluation results
            eval_path = os.path.join(
                self.config.experiment.output_dir,
                'evaluation_results.json'
            )
            with open(eval_path, 'w') as f:
                # Convert to JSON-serializable format
                json_results = {
                    'metrics': eval_results['metrics'],
                    'calibration': {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in eval_results['calibration'].items()
                    },
                    'failure_analysis': eval_results['failure_analysis']
                }
                json.dump(json_results, f, indent=2)
            
            print(f"\nSaved evaluation results to {eval_path}")
    
    def run_profiling(self, test_loader):
        """Step 6: Performance profiling (30 min)"""
        print("\n" + "="*80)
        print("STEP 6: PERFORMANCE PROFILING")
        print("="*80)
        
        if not self.config.experiment.profile:
            print("Profiling disabled in config")
            return
        
        with profile_section("Performance Profiling", self.gpu_monitor):
            # Load CLIP model for benchmarking
            clip_retrieval = CLIPRetrieval(self.config)
            
            # Benchmark inference
            throughput_data = benchmark_inference(
                clip_retrieval.model,
                test_loader,
                num_batches=20,
                warmup_batches=5,
                device=self.config.device
            )
            
            # Save GPU snapshots
            snapshot_path = os.path.join(
                self.config.experiment.output_dir,
                'gpu_snapshots.json'
            )
            self.gpu_monitor.save_snapshots(snapshot_path)
            self.gpu_monitor.print_summary()
            
            # Create performance report
            report_path = os.path.join(
                self.config.experiment.output_dir,
                'performance_report.json'
            )
            create_performance_report(
                self.gpu_monitor.snapshots,
                throughput_data,
                self.training_times,
                self.config,
                report_path
            )
    
    def create_final_report(self):
        """Step 7: Create final results summary (15-30 min)"""
        print("\n" + "="*80)
        print("STEP 7: FINAL REPORT GENERATION")
        print("="*80)
        
        report = {
            'experiment_info': {
                'hardware': self.config.hardware.device_type,
                'num_seeds': self.config.experiment.num_seeds,
                'seeds': self.config.experiment.seeds,
                'dataset': self.config.data.dataset_name,
                'num_samples': self.config.data.num_samples,
            },
            'results_summary': {},
            'training_times': self.training_times,
            'total_time_hours': sum(self.training_times.values()) / 3600
        }
        
        # CLIP Baseline
        if self.results['clip_baseline']:
            test_results = [r['test_metrics'] for r in self.results['clip_baseline']]
            report['results_summary']['clip_baseline'] = {
                'recall@1': {
                    'mean': float(np.mean([r['recall@1'] for r in test_results])),
                    'std': float(np.std([r['recall@1'] for r in test_results]))
                },
                'recall@5': {
                    'mean': float(np.mean([r['recall@5'] for r in test_results])),
                    'std': float(np.std([r['recall@5'] for r in test_results]))
                }
            }
        
        # Linear Probe
        if self.results['linear_probe']:
            test_results = [r['test_metrics'] for r in self.results['linear_probe']]
            report['results_summary']['linear_probe'] = {
                'recall@1': {
                    'mean': float(np.mean([r['recall@1'] for r in test_results])),
                    'std': float(np.std([r['recall@1'] for r in test_results]))
                },
                'recall@5': {
                    'mean': float(np.mean([r['recall@5'] for r in test_results])),
                    'std': float(np.std([r['recall@5'] for r in test_results]))
                }
            }
        
        # Captioner
        if self.results['captioner_full']:
            full_ciders = []
            for r in self.results['captioner_full']:
                gen = r['test_metrics']['generated_captions']
                ref = r['test_metrics']['reference_captions']
                full_ciders.append(compute_cider_score(gen, ref))
            
            report['results_summary']['captioner_full'] = {
                'cider': {
                    'mean': float(np.mean(full_ciders)),
                    'std': float(np.std(full_ciders))
                }
            }
        
        if self.results['captioner_lora']:
            lora_ciders = []
            for r in self.results['captioner_lora']:
                gen = r['test_metrics']['generated_captions']
                ref = r['test_metrics']['reference_captions']
                lora_ciders.append(compute_cider_score(gen, ref))
            
            report['results_summary']['captioner_lora'] = {
                'cider': {
                    'mean': float(np.mean(lora_ciders)),
                    'std': float(np.std(lora_ciders))
                }
            }
        
        # Save report
        report_path = os.path.join(
            self.config.experiment.output_dir,
            'results.json'
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nDataset: {report['experiment_info']['dataset']}")
        print(f"Hardware: {report['experiment_info']['hardware']}")
        print(f"Total Time: {report['total_time_hours']:.2f} hours")
        
        print("\n" + "-"*60)
        print("Retrieval Performance (Test Set)")
        print("-"*60)
        
        if 'clip_baseline' in report['results_summary']:
            cb = report['results_summary']['clip_baseline']
            print(f"\nCLIP Baseline:")
            print(f"  R@1: {cb['recall@1']['mean']:.2f} ± {cb['recall@1']['std']:.2f}%")
            print(f"  R@5: {cb['recall@5']['mean']:.2f} ± {cb['recall@5']['std']:.2f}%")
        
        if 'linear_probe' in report['results_summary']:
            lp = report['results_summary']['linear_probe']
            print(f"\nLinear Probe:")
            print(f"  R@1: {lp['recall@1']['mean']:.2f} ± {lp['recall@1']['std']:.2f}%")
            print(f"  R@5: {lp['recall@5']['mean']:.2f} ± {lp['recall@5']['std']:.2f}%")
            
            if 'clip_baseline' in report['results_summary']:
                improvement = lp['recall@1']['mean'] - cb['recall@1']['mean']
                print(f"  Improvement over baseline: +{improvement:.2f}%")
        
        print("\n" + "-"*60)
        print("Captioning Performance (Test Set)")
        print("-"*60)
        
        if 'captioner_full' in report['results_summary']:
            cf = report['results_summary']['captioner_full']
            print(f"\nFull Fine-tuning:")
            print(f"  CIDEr: {cf['cider']['mean']:.2f} ± {cf['cider']['std']:.2f}")
        
        if 'captioner_lora' in report['results_summary']:
            cl = report['results_summary']['captioner_lora']
            print(f"\nLoRA Fine-tuning:")
            print(f"  CIDEr: {cl['cider']['mean']:.2f} ± {cl['cider']['std']:.2f}")
        
        print(f"\n✓ Full report saved to: {report_path}")
        print(f"✓ All outputs saved to: {self.config.experiment.output_dir}")
        
        return report
    
    def run_full_pipeline(self):
        """Run complete pipeline"""
        overall_start = time.time()
        
        print("\n" + "="*80)
        print("STARTING FULL PIPELINE")
        print("="*80)
        print(f"Configuration: {self.config}")
        print(f"Seeds: {self.config.experiment.seeds}")
        
        try:
            # Step 1: Data preparation
            (train_loader, val_loader, test_loader,
             train_data, val_data, test_data) = self.run_data_preparation()
            
            # Step 2: CLIP baseline
            self.run_clip_baseline_all_seeds(train_loader, val_loader, test_loader)
            
            # Step 3: Linear probe
            self.run_linear_probe_all_seeds()
            
            # Step 4: Captioner
            self.run_captioner_all_seeds(train_data, val_data, test_data)
            
            # Step 5: Calibration and OOD
            self.run_calibration_and_ood_analysis(test_loader)
            
            # Step 6: Profiling
            self.run_profiling(test_loader)
            
            # Step 7: Final report
            final_report = self.create_final_report()
            
            overall_elapsed = time.time() - overall_start
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Total time: {overall_elapsed:.2f}s ({overall_elapsed/3600:.2f} hours)")
            
            return final_report
            
        except Exception as e:
            print(f"\n✗ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    parser = argparse.ArgumentParser(description="Run Retrieval + Captioning Pipeline")
    parser.add_argument('--device', type=str, default='rtx5000_ada',
                       choices=['rtx5000_ada', 'a5000', 'rtx3070'],
                       help='Hardware configuration')
    parser.add_argument('--dataset', type=str, default='coco',
                       choices=['coco', 'flickr30k'],
                       help='Dataset to use')
    parser.add_argument('--no-profile', action='store_true',
                       help='Disable profiling')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(device_type=args.device)
    config.data.dataset_name = args.dataset
    config.experiment.profile = not args.no_profile
    config.experiment.output_dir = args.output_dir
    
    # Print hardware info
    config.print_hardware_info()
    
    # Run pipeline
    runner = PipelineRunner(config)
    runner.run_full_pipeline()

if __name__ == "__main__":
    main()
