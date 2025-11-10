# 🚀 YOLO Segmentation Telegram Bot in Python  
### Multi-Model AI Image Segmentation | Brain Tumor, Roads, Cracks, Leaf Disease, Person & Pothole Segmentation

This repository contains a **Telegram Segmentation Bot** built using **Python**, **YOLO segmentation models**, and the **python-telegram-bot** library.  
The bot supports **multiple segmentation models**, allowing users to select a model and receive high-quality segmented images directly inside Telegram.

---

## ✅ Features
- Multi-model segmentation support  
- YOLO-based segmentation using ultralytics  
- InlineKeyboard model selection  
- Overlay masks with alpha blending  
- Clean and modular codebase  
- Real-time inference using Telegram bot API  
- Easy to deploy and extend  

---

## 🧠 Supported Segmentation Models
| Model ID | Task | Weights File | Classes |
|---------|------|--------------|---------|
| 1 | Brain Tumor Segmentation | `brain_tumor.pt` | bg, Tumor |
| 2 | Road Segmentation | `road.pt` | bg, Road |
| 3 | Crack Detection | `cracks.pt` | bg, Cracks |
| 4 | Leaf Disease Segmentation | `leaf_disease.pt` | bg, Disease |
| 5 | Person Segmentation | `person.pt` | bg, Person |
| 6 | Pothole Detection | `pothole.pt` | bg, Pothole |

---

## 📥 Download Model Weights  
Place all weights inside the `Weights/` folder.
https://drive.google.com/drive/folders/19ObW9wy7dKRTJfxgX4gCLxO-6hXRKOfy?usp=sharing
