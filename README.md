# YOSS: You Only Speak Once to See for Audio Grounding

**Authors**: Wenhao Yang, Jianguo Wei, Wenhuan Lu, Lei Li  
**PDF**: [arXiv:2409.18372v2](https://arxiv.org/abs/2409.18372v2)  
**Conference**: [ICASSP 2025](https://2025.ieeeicassp.org/) (International Conference on Acoustics, Speech and Signal Processing)  

---  

## 🌟 Introduction  
YOSS (You Only Speak Once to See) is an innovative framework for **Audio Grounding**, enabling robots and computer vision systems to localize objects in images using spoken language. By integrating pre-trained audio models (e.g., HuBERT) with visual models (e.g., CLIP) via contrastive learning and multi-modal alignment, YOSS bridges the gap between speech commands/descriptions and visual object localization. This repository provides the implementation of YOSS, which achieves robust performance on open-vocabulary object detection tasks using audio prompts.



## 📖 Project Overview  
### Key Features  
- **Audio-Visual Contrastive Learning**: Aligns audio and image embeddings in a shared semantic space using pre-trained models like CLIP and HuBERT.  
- **Multi-Modal Grounding**: Combines multi-scale visual features with audio embeddings to generate object bounding boxes via a YOLOv8-based detection backbone.  
- **Open-Vocabulary Support**: Works seamlessly with unseen object classes by leveraging pre-trained cross-modal representations.  

### Core Contributions  
1. Propose the **Audio Grounding** task, enabling object localization from speech inputs.  
2. Develop a framework that fuses audio-image contrastive learning and YOLO-based detection for end-to-end grounding.  
3. Validate effectiveness on datasets like Flickr, COCO, and GQA, demonstrating competitive performance in audio-guided object detection.  
   

## Getting started

### How to Install
```bash

### pytorch environment
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

### SpeechCLIP 
cd thirds/SpeechCLIP
pip install -r requirements.txt

### YOLO-World
cd ../thirds/YOLO-World
pip install -e .

# pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
# pip install omegaconf==2.0.3
```


## 📊 Datasets & Preprocessing  
### Datasets Used  
| Dataset       | Role                                      | Notes                                      |  
|---------------|-------------------------------------------|------------------------------------------|  
| **Flickr 8K**  | Audio-Image Contrastive Learning          | Contains 8k images with 40k audio captions.|  
| **Flickr 30K** | Audio Grounding Training                  | 31k images with 158k audio-grounding pairs.|  
| **COCO2014**   | Validation                                | Standard object detection benchmark.      |  
| **SpokenCOCO** | Speech Synthesis Baseline                 | Audio versions of MSCOCO captions.        |  

### Data Preparation  
1. **Audio Synthesis** (for text-to-speech):  
   - Use `SpeechT5` to generate audio captions from text annotations.  
   - Filter low-quality audio using `Whisper` ASR to ensure alignment accuracy.  

2. **Pseudo Labeling**:  
   - Generate bounding box annotations for audio captions using pre-trained open-vocabulary detectors (e.g., GLIP) for weakly supervised training.  
   - Time stampes for annotations are annotated by whisper_timestamps models.  
  
```bash
wget https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads/flickr_audio.tar.gz -P data/flickr
tar -xzvf data/flickr/flickr_audio.tar.gz -C data/flickr

mkdir data/flickr/Images
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -P data/flickr
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -P data/flickr
unzip data/flickr/Flickr8k_text.zip -d  data/flickr
unzip data/flickr/Flickr8k_Dataset.zip -d  data/flickr/Images
```

### Pretrained Models
1. **YOLO-World**:Text Grounding Model
    - yolo_world_v2_s [HF Checkpoints 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth)
2. **HuBERT**
3. **CLIP**
4. **YOSS**
    - YOSS-S without finetuning [HF Checkpoints 🤗](https://huggingface.co/blackstone/YOSS-S)

## 🚀 Usage  
### 1. Audio-Visual Contrastive Pre-Training  
Train the audio-image alignment model using Flickr 8K:  

```bash
cd thirds/SpeechCLIP
bash egs/hubert/train.sh
```

- Aligns audio embeddings (HuBERT) and image embeddings (CLIP) using contrastive loss.  
- Outputs a shared semantic space for cross-modal retrieval.  

### 2. Audio Grounding within YOSS-S 
Fine-tune the grounding model on Flickr 30K and validate on COCO2014:  
```bash
cd thirds/YOLO-World
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --master_port=41705 --nnodes=1 tools/train.py configs/pretrain/YOSS_S.py --launcher pytorch --amp
```

- Uses YOLOv8's CSPDarknet backbone and NAS-FPN for multi-scale feature fusion.  
- Combines classification loss (CrossEntropy) and localization loss (Distribution Focal Loss + IoU Loss).  

### 3. Inference  
Detect objects from audio prompts in real-time:  
```python  
python inference.py \  
    --image-path input_image.jpg \  
    --audio-path "person riding a bicycle"_speech.wav \  
    --model-ckpt grounding_ckpt.pth  
```
- Input: Speech waveform (e.g., ".wav") and RGB image.  
- Output: Bounding boxes with class confidence scores aligned to the audio prompt.  

## 📈 Results  
### Key Performance Metrics  
| Task                | Dataset       | AP (COCO-style) | AP@50 | AP@75 |  
|---------------------|---------------|----------------|-------|-------|  
| Audio Object Detection | COCO2017      | 39.2           | 53.3  | 42.6  |  
| Zero-Shot Detection  | LVIS Minival  | 16.3           | -     | -     |  

- YOSS outperforms baseline audio-image retrieval methods (e.g., SpeechCLIP) by 10% in R@10 on Flickr 8K.  
- Demonstrates strong generalization to unseen audio classes via pre-trained cross-modal embeddings.  


## 📚 Citation  
If you find this work useful, please cite our paper:  
```bibtex
@INPROCEEDINGS{10889085,
  author={Yang, Wenhao and Wei, Jianguo and Lu, Wenhuan and Li, Lei},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={You Only Speak Once to See}, 
  year={2025},
  keywords={YOLO;Visualization;Computer vision;Grounding;Computational modeling;Contrastive learning;Signal processing;Text to speech;Object recognition;Speech processing;Audio Grounding;Multi-modal;Detection},
  doi={10.1109/ICASSP49660.2025.10889085}}
```


## 🔗 Acknowledgement  
- **SpeechCLIP**: [GitHub](https://github.com/atosystem/SpeechCLIP) (Audio-Text-Image contrastive learning)  
- **YOLO-World**: [GitHub](https://github.com/AILab-CVC/YOLO-World) (Open-vocabulary detection backbone)  
- **whisper-timestamped**: [GitHub](https://github.com/linto-ai/whisper-timestamped) (Word-level timestamps and confidence Generation)  


## License

YOSS is under the Apache license 2.0.