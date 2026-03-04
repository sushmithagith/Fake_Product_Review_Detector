# Fake Review Detector - Test Examples

Use these examples to test the Fake Review Detector model:

---

## 🔴 FAKE Reviews (Label: 1 - Should be detected as FAKE)

### Example 1: Short Generic Review
```
Five Stars,Great fit!
```
**Why Fake:** Very short, only star rating, no actual details

### Example 2: Promotional Language
```
BEST PRODUCT EVER! You MUST buy now! Limited offer, guaranteed 100% working! Life changing! Buy now!
```
**Why Fake:** Excessive exclamation marks, pressure tactics, unrealisitic claims

### Example 3: Generic Star Rating Only
```
Three Stars,A bit large.
```
**Why Fake:** Just a star rating with minimal text

### Example 4: With Star Symbols
```
★ THESE REALLY DO WORK GREAT WITH SOME TWEAKING ★
```
**Why Fake:** Contains star symbols, repetitive

### Example 5: Mentioning Discount (Often Fake)
```
Five stars i purchased this product at a discounted price in exchange for my honest review
```
**Why Fake:** Disclosure of receiving discount for review

### Example 6: Very Short
```
Five Stars,Perfect
```
**Why Fake:** Too short to be a genuine review

---

## 🟢 GENUINE Reviews (Label: 0 - Should be detected as GENUINE)

### Example 1: Detailed Personal Experience
```
I love this dress. Absolute favorite for winter. Heavy material. Stretchy, shows shape well. I am 5ft 7, 120 lbs. Ordered 2-8. Fits fine. Not tight at all. But not to loose. Very comfortable. I live in ND, and it keeps you warm during winter.
```
**Why Genuine:** Detailed, specific measurements, personal experience

### Example 2: Balanced Review (Pros & Cons)
```
The neck on this is a bit saggy, but is a nice sweatshirt.
```
**Why Genuine:** Balanced, mentions both positive and negative

### Example 3: Specific Usage Details
```
I've been using Mine For a Few Years Now. First, I Paid a Few Dollars Less For Mine and The Price Has Jumped. They're All Imported, so Try to Find Either a Cheaper One or One That's Extremely Well Made.
```
**Why Genuine:** Specific details, personal experience over time

### Example 4: Natural Writing Style
```
Nice socks, great colors, just enough support for wearing with a good pair of sneakers.
```
**Why Genuine:** Natural, descriptive, not overly promotional

### Example 5: Detailed Negative Review
```
Shirt a bit too long, with heavy hem, which inhibits turning over. I cut off the bottom two inches all around, and am now somewhat comfortable. Overall, material is a bit too heavy for my liking.
```
**Why Genuine:** Detailed explanation of issues

### Example 6: Normal Positive Review
```
It arrived quickly and looks just as image. Very comfortable!
```
**Why Genuine:** Natural, not overly promotional

---

## Quick Test in App

The Streamlit app has these examples built into the dropdown menu:
1. Go to the app
2. Find "Try an example" dropdown
3. Select "Promotional Fake", "Generic Fake", or "Genuine Review"
4. Click "Analyze Review"

---

## Expected Results

| Example | Expected Prediction | Key Indicators |
|---------|-------------------|----------------|
| "Five Stars,Great fit!" | FAKE | Very short, no details |
| "BEST PRODUCT EVER! Buy now!" | FAKE | Excessive !, pressure tactics |
| Detailed personal review | GENUINE | Specific details, natural writing |
