---
name: word-of-the-day
description: Fetch today's Word of the Day and design a toddler-friendly activity to teach it.
argument-hint: Optional age group, e.g. "ages 3 to 4"
agent: Idea Generator
tools: ['fetch/*']
---

Follow the [toddler-friendly instructions](../instructions/toddler-friendly.instructions.md) in
every reply. Build the activity around [our toys](../instructions/our-toys.instructions.md), and
don't ask me which toys we have.

Use the fetch tool to get https://wordaday.kids/word-of-the-day-for-kids/ and find today's word,
with its meaning, synonyms, antonym, phrase and "finish the story" starter.

Then help me design a short, playful activity that teaches today's word to toddlers. Use the age
group I gave after the command, if any; otherwise, aim for ages 2 to 4.

Start by telling me today's word and what it means, in toddler-friendly words. Then ask me one
question at a time to shape the activity. When we have enough, summarize the activity as
simple step-by-step instructions a parent or teacher could follow.
