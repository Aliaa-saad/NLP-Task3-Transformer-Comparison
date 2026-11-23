export interface Example {
  id: string;
  label: string;
  text: string;
}

const LONG_REVIEW = [
  "We booked this place for my parents' anniversary dinner after reading a lot of reviews, and I want to give a full picture because the evening had real highs and lows.",
  "Parking was a nightmare: the lot was full, the valet stand was unattended for ten minutes, and we ended up circling the block three times before finding a spot on a side street.",
  "When we finally walked in, the hostess was apologetic and seated us right away even though we were fifteen minutes late for our reservation, which I appreciated.",
  "The dining room is gorgeous, with warm lighting, exposed brick, and enough space between tables that you can actually hear each other talk.",
  "Our server, Marco, was attentive without hovering, knew the wine list inside and out, and recommended a Rioja that paired perfectly with the lamb.",
  "Appetizers were mixed. The burrata with roasted peaches was outstanding, creamy and bright, but the calamari was rubbery and under-seasoned, and we left half of it.",
  "For mains, my dad had the braised short rib, which fell apart with a fork and came with silky mashed potatoes. My mom's sea bass was cooked well but the sauce was overly salty.",
  "I ordered the mushroom risotto, which was rich and earthy, though the portion was smaller than I expected for the price.",
  "There was a long gap between the appetizers and the mains, close to forty minutes, and nobody checked in during that time except to refill water.",
  "Dessert redeemed a lot. The kitchen sent out a complimentary tiramisu with a candle for the anniversary, which was a sweet touch, and the chocolate torte was decadent.",
  "The bill was steep, and I noticed an automatic service charge that wasn't mentioned anywhere on the menu, which felt a little sneaky.",
  "By the end of the night, though, my parents were genuinely happy, the staff were warm, and the last few dishes left us all smiling.",
  "Overall, despite the parking chaos, the slow kitchen, and the surprise fee, I would absolutely come back for the food and the service. Highly recommended for special occasions!",
].join(" ");

export const EXAMPLES: Example[] = [
  {
    id: "positive",
    label: "Clearly positive",
    text: "Absolutely loved this place! The pasta was fresh and perfectly cooked, the staff were incredibly friendly, and the tiramisu was the best I've ever had. We'll definitely be back next week.",
  },
  {
    id: "negative",
    label: "Clearly negative",
    text: "Worst dining experience in years. We waited over an hour for cold, greasy food, the waiter was rude when we complained, and they still charged us full price. Never coming back.",
  },
  {
    id: "neutral",
    label: "Neutral / mixed",
    text: "It was okay. The burger was decent and the fries were fine, but nothing really stood out. Service was a bit slow, prices are average. Not bad, not great — I might come back if I'm in the area.",
  },
  {
    id: "sarcastic",
    label: "Sarcastic",
    text: "Oh wonderful, another 45-minute wait for a 'quick' lunch. I just love paying $18 for a lukewarm sandwich and getting my order wrong twice. Truly a five-star experience. Can't wait to never do this again.",
  },
  {
    id: "long",
    label: "Long (gets truncated)",
    text: LONG_REVIEW,
  },
];

/** Short hand-written reviews (not from the dataset) for the hero's live ticker. */
export const TICKER_REVIEWS: string[] = [
  "Best ramen in town. Rich broth, perfect noodles, friendly staff.",
  "Waited 40 minutes and they still got my order wrong.",
  "Decent coffee, nothing special. Prices are fair enough.",
  "The team went above and beyond for our anniversary dinner!",
  "Room was clean, but the AC rattled all night long.",
  "Never again. Cold food, rude manager, sticky tables.",
];
