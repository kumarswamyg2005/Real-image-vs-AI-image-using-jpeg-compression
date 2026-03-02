# Presentation Script - Detecting AI Images Through JPEG Compression Forensics

## 🎤 Opening (Slide 1)

"Good morning/afternoon everyone. Today I'm presenting a project on detecting AI images through JPEG compression forensics. Our system can tell if an image is real or AI-generated with 97.2% accuracy. We trained this model on 348,000 images to solve a very real problem in today's digital world."

---

## Slide 2: What Is This Project?

"So what's the problem we're solving? With AI generators like Stable Diffusion, DALL-E, and Midjourney becoming so advanced, they can now create images that look completely real to the human eye. Whether it's a photo of a person, a landscape, or any object - AI can generate it perfectly.

The challenge is: how do we tell them apart? That's what this project addresses. We built a detector using JPEG compression forensics that can identify whether an image came from a real camera or was AI-generated."

---

## Slide 3: How We Use JPEG Compression

"Let me explain how JPEG compression helps us detect AI images.

When a real camera takes a photo, it goes through a specific process:

- Light hits the camera sensor - this is a physical process
- The camera converts this to pixels
- Then JPEG compression kicks in - it breaks the image into 8×8 pixel blocks
- It uses mathematical transforms called DCT to compress these blocks
- This creates unique 'fingerprints' that are like a signature from the camera

We extract 70 different forensic features from this process:

- DCT coefficients - these are the compression math patterns
- Blocking artifacts - discontinuities at those 8×8 block boundaries
- Benford's Law - a natural statistical pattern that real data follows
- Camera dimension signatures - like 12 megapixels, 4:3 aspect ratio

The key insight is: AI-generated images have different compression patterns because they weren't created by a physical camera sensor. They're created by neural networks, so the math is different."

---

## Slide 4: What About PNG Images?

"Now you might be wondering - what about PNG images?

Our model works best with JPEG because JPEG uses lossy compression - it loses some data to make files smaller, and this creates the artifacts and patterns we can detect.

PNG, on the other hand, uses lossless compression - it maintains perfect quality but doesn't create those compression artifacts we're looking for.

Our model can still analyze PNG images by checking things like dimensions and resolution patterns, but the accuracy may be lower because we trained specifically on 348,000 JPEG images. For best results, use JPEG images."

---

## Slide 5: Our Approach

"Let me tell you about our journey. We initially tried the traditional deep learning approach using ResNet50 - this is a neural network that looks at the pixels and tries to learn 'what AI images look like.'

But it failed catastrophically. It gave us 100% false positives - meaning every single real photo was being marked as AI-generated. The model learned superficial pixel patterns that don't generalize well.

So we completely changed our approach. Instead of asking 'does this LOOK like AI?', we asked 'how was this created?' We switched to forensic analysis - examining the compression signatures.

We use a Random Forest classifier - it's a machine learning model that analyzes those 70 forensic features we extracted. The result? We went from 100% false positives down to just 2.6%, and achieved 97.2% overall accuracy.

The key insight here is: AI can make images that LOOK perfectly real to our eyes, but it can't easily fake the mathematical patterns from real camera compression. That's physics, and physics is hard to fake."

---

## Slide 6: Training Process

"Now let me talk about how we trained this model.

We used 348,385 images in total - that's a large-scale dataset. Half were real photos, half were AI-generated to keep it balanced.

- 120,000 images came from the CIFAKE dataset - these are objects, scenes, landscapes
- 140,000 came from a faces dataset - human faces, portraits

For each image, we extract those 70 forensic features I mentioned. Then we train a Random Forest classifier - it's essentially 300 decision trees that work together and vote on the final classification.

The training took 6 to 8 hours on a regular Mac computer - no GPU required, just CPU. This makes it very practical and accessible."

---

## Slide 7: Results

"So what were our results?

97.2% overall accuracy. Let me break that down:

- 97.4% accuracy on real photos - so we correctly identify real photos 97.4% of the time
- 97.0% accuracy on AI images - we correctly catch AI images 97% of the time
- Only 2.6% false positive rate - that means only 2.6% of real photos get wrongly flagged as AI

And remember, this all runs on CPU only - no GPU needed - and works completely offline. It's a practical, deployable system."

---

## Slide 8: Web Application Demo

"We didn't just build a model - we built a working web application that anyone can use.

Here's how it works:

1. You upload an image through a simple drag-and-drop interface
2. The system analyzes the JPEG compression patterns
3. It checks the camera dimension signatures
4. Then it shows you the result - either 'Real Photo' with a camera emoji, or 'AI-Generated' with a robot emoji
5. You also get a confidence score from 0 to 100%

It runs locally on your computer at localhost port 5001, and it processes images in under 1 second. It's fast, it's accurate, and it's practical."

[If you can show a demo here, do it!]

---

## Slide 9: Future Improvements

"Looking ahead, there are several ways we can improve this system:

First, we can improve PNG detection by training specifically on PNG images and developing PNG-specific forensic features, since PNG doesn't have JPEG compression artifacts.

Second, we can identify which specific AI generator created an image - Stable Diffusion, DALL-E, and Midjourney each have unique fingerprints, so we could build a classifier that tells you not just IF it's AI, but WHICH AI made it.

Third, we could build mobile apps for iOS and Android so people can check images on the go before sharing them on social media.

And fourth, we're targeting even higher accuracy - aiming for 99%+ with less than 1% false positives by adding more forensic features and training on even larger datasets."

---

## Slide 10: Summary

"To summarize everything:

We built an AI image detector using JPEG compression forensics - analyzing how images are created, not just what they look like.

We achieved 97.2% accuracy and fixed the false positive problem that plagued deep learning approaches - going from 100% false positives down to just 2.6%.

The system works on 348,000 images, runs on CPU only, works offline, and we have a working web demo you can use right now.

The key takeaway is this: Forensic analysis beats deep learning for this task. Instead of looking at pixels and guessing, we check the compression math - and that's much harder to fake.

Thank you. I'm happy to take any questions."

---

## 💡 If Asked Common Questions:

### "Why not use EXIF metadata?"

"Great question. EXIF metadata can be easily faked - anyone can copy EXIF from a real photo and paste it onto an AI image in just 2 lines of code. Also, social media platforms like Facebook and Instagram automatically strip EXIF data for privacy reasons. So we can't rely on it. Compression patterns are mathematical signatures that are much harder to fake."

### "What if AI gets better at faking compression?"

"That's a valid concern. But to fake compression patterns, AI would need to understand and replicate the exact physics of how camera sensors and JPEG compression work at a mathematical level. That's much harder than just making realistic-looking pixels. We're betting that physics is harder to fake than appearances - and so far, that's proven true."

### "How fast does it work?"

"Very fast - under 1 second per image. Random Forest inference is quick because it's just running through 300 decision trees. We extract the features once, then classification is almost instant. No GPU needed."

### "Can it detect video deepfakes?"

"Not yet - that's definitely on our roadmap for future work. But the same forensic principles would apply. We could analyze compression patterns frame by frame in videos. That's a natural next step."

### "What about images edited in Photoshop?"

"Interesting question. Edited images would show signs of double compression - being saved multiple times - which is one of our 70 forensic features. So heavy edits might get flagged, but light edits on real photos would likely still be detected as real since the underlying camera fingerprint is still there."

---

## 🎯 Closing Tips:

- **Speak confidently** - You built something that works!
- **Make eye contact** with your evaluator
- **Don't rush** - pause between slides
- **If you don't know something** - say "That's an excellent question for future research" instead of guessing
- **Show enthusiasm** - you solved a real problem!

**You've got this! Good luck! 🎉**
