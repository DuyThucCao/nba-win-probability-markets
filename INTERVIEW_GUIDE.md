# Interview guide: understand this project before you list it

Run `python run_all.py` and the tests yourself. Then make sure you can
answer these questions in your own words.

1. **What is the target?** The probability that the home team wins. It is
   not the point spread, the total, or a betting return.

2. **What does "removing the vig" mean?** Implied probabilities from both
   moneylines sum to more than 1. In this sample they sum to about 1.038,
   which is the bookmaker's margin. Dividing each by the sum gives a fair
   probability.
   - Example: -150 gives 0.600 and +130 gives 0.435. The sum is 1.035, so
     the no-vig home probability is 0.580.

3. **Why walk-forward instead of a random split?** A random split lets later
   seasons inform earlier predictions. Here every season is predicted only
   from earlier seasons. The tests check that changing future results cannot
   change earlier predictions.

4. **How does the Elo update work?**
   - Expected win probability = 1 / (1 + 10^(-(diff + HFA) / 400)).
   - After the game, the rating moves by K × (margin multiplier) × (result − expected).
   - The margin multiplier stops blowouts by heavy favorites from moving
     ratings too much.

5. **What is a forecast encompassing test?** Regress the outcome on the
   log-odds of both forecasts. If the market price contains everything the
   model knows, the model's coefficient is zero. Here it is -0.001
   (p = 0.98).

6. **Why was the stacked model fit on out-of-fold predictions?** In-sample
   predictions are overconfident on the data they were fit on. The meta-model
   would then learn to trust them too much.

7. **What exactly did the line-movement test show?**
   - The model's disagreement with the opening spread is **associated** with
     the later move toward the close.
   - It does **not** show alpha or a tradable strategy. There are no
     opening-line timestamps, no opening moneylines, and no account of
     limits or fees.
   - The rested-games check exists because an opening line can be posted
     before a team's previous game ends.

8. **What data problems did you find?** Twelve wrong dates and twelve
   mistyped scores in the odds archive. All were caught by matching every
   game to the Basketball-Reference schedule. A team listed twice on the same
   date is what exposed the problem.

9. **What would you do next?** Add injury and lineup data, test on seasons
   after 2023, collect timestamped prediction-market prices (for example
   exchange contracts), and model fees and liquidity.

Do not describe this project as a profitable strategy, professional sports
data work, or work done without assistance.
