# Advantek Laboratories — Technical Assignment

Welcome, and thanks for your interest.

Advantek builds decision models for energy and industrial clients. Most of our work looks like this: a client has data, a decision to make, and no clear path between the two. Our job is to build the path — quickly, and honestly enough that they can act on it.

This assignment is a small version of that.

---

## The situation

A Hungarian energy retailer buys electricity on the day-ahead market. Every day before noon, they must commit to how much power they will need for **each hour of tomorrow**. If they buy too little, they cover the shortfall on the balancing market at a penalty. If they buy too much, they sell the excess back — usually at a loss.

They currently do this by hand, using a spreadsheet and experience.

They have asked us whether it can be done better.

## What we'd like from you

Something we could run tomorrow morning that produces tomorrow's hourly demand forecast, plus a short write-up of what you did and why.

That's the whole brief. It is underspecified on purpose — deciding what "better" means here, and what's worth building in the time available, is part of what we're asking you to do.

---

## Data

**Historical demand** — `data/hu_load_15min.csv` in this repo. Quarter-hourly net electricity demand for Hungary, several years, from the Hungarian transmission system operator, alongside their own published day-ahead forecast for the same series.

**Everything else is up to you.** If you want other inputs, find them. One note to save you time: if you go looking for weather data, [Open-Meteo](https://open-meteo.com/) is free and needs no API key or registration — read their docs properly, there is more there than is obvious at first glance.

Please don't use any source that requires a paid subscription. We want you spending the time on the problem, not on procurement.

---

## The write-up

Keep it short — one page is fine. Cover:

1. **What you decided the question actually was.** The brief above is vague. What did you narrow it to, and what did you rule out?
2. **What you'd tell the client.** Should they use this? What would it get them? What would you want them to understand about its limitations before they trusted it with money?
3. **What you don't trust.** In the data, in your approach, in your own results.
4. **What you'd check first if you had another day on this.** Be specific — not "more feature engineering," but the actual thing that's bothering you.

This document matters as much as the code. We will spend most of the interview on it.

---

## Time

Budget around **four hours**. That is guidance, not a rule — if you want to spend six, spend six; if you're done in three and happy with it, stop.

Four hours is not enough to do everything you might want to do. Deciding what to leave out is part of the exercise, and we'd rather see something small and defensible than something broad and unexamined. If there are things you knowingly skipped, say so.

---

## AI tools

Use them. We do, on real client work, every day. There are no restrictions and no bonus points for not using them.

Please tell us in the write-up which tools you used. This is not a trap — we're calibrating, and if you produced good work on a weaker model we'd like to know that.

---

## Submission

Fork this repository, do the work, and open a pull request. If GitHub is a problem for any reason, a `.zip` by email is fine.

---

## What happens next

If we invite you to interview, we will go through your submission in detail and ask you to explain and defend your choices — including the small ones, and including the things you didn't do. We'll also ask you to extend your reasoning to a related problem you haven't seen.

We are not checking whether you wrote the code yourself. We're checking whether you understand what you handed us and can tell us why it's built that way. That's the actual skill, and it's what the rest of the process is designed to find.

Good luck.
