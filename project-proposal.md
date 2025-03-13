# GRPO For Optimization Problems
## Pipeline Overview
### Dataset
- [TSLib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
- [TSLib Overview](https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html)
### Sample Problem
City Coordinates
```
1 6734 1453
2 2233 10
3 5530 1424
4 401 841
5 3082 1644
6 7608 4458
7 7573 3716
8 7265 1268
9 6898 1885
10 1112 2049
11 5468 2606
12 5989 2873
13 4706 2674
14 4612 2035
15 6347 2683
16 6107 669
17 7611 5184
18 7462 3590
19 7732 4723
20 5900 3561
21 4483 3369
22 6101 1110
23 5199 2182
24 1633 2809
25 4307 2322
26 675 1006
27 7555 4819
28 7541 3981
29 3177 756
30 7352 4506
31 7545 2801
32 3245 3305
33 6426 3173
34 4608 1198
35 23 2216
36 7248 3779
37 7762 4595
38 7392 2244
39 3484 2829
40 6271 2135
41 4985 140
42 1916 1569
43 7280 4899
44 7509 3239
45 10 2676
46 6807 2993
47 5185 3258
48 3023 1942
```
Intercity Distance Table
- [Distance Table](https://people.sc.fsu.edu/~jburkardt/datasets/tsp/att48_d.txt)  

Optimal Tour
```
1
8
38
31
44
18
7
28
6
37
19
27
17
43
30
36
46
33
20
47
21
32
39
48
5
42
24
10
45
35
4
26
2
29
34
41
16
22
3
23
14
25
13
11
12
15
40
9
1
```
Optimal Tour Distance
```
33523
```
### Model Input
Used Claude 3.7 for this example.  
The following is a list of coordinates for x cities. Solve TSP:  
```
1 6734 1453
2 2233 10
3 5530 1424
4 401 841
5 3082 1644
6 7608 4458
7 7573 3716
8 7265 1268
9 6898 1885
10 1112 2049
11 5468 2606
12 5989 2873
13 4706 2674
14 4612 2035
15 6347 2683
16 6107 669
17 7611 5184
18 7462 3590
19 7732 4723
20 5900 3561
21 4483 3369
22 6101 1110
23 5199 2182
24 1633 2809
25 4307 2322
26 675 1006
27 7555 4819
28 7541 3981
29 3177 756
30 7352 4506
31 7545 2801
32 3245 3305
33 6426 3173
34 4608 1198
35 23 2216
36 7248 3779
37 7762 4595
38 7392 2244
39 3484 2829
40 6271 2135
41 4985 140
42 1916 1569
43 7280 4899
44 7509 3239
45 10 2676
46 6807 2993
47 5185 3258
48 3023 1942
```
### Model Output
Code:
```
// Parse the input data
const citiesData = `
1 6734 1453
2 2233 10
3 5530 1424
4 401 841
5 3082 1644
6 7608 4458
7 7573 3716
8 7265 1268
9 6898 1885
10 1112 2049
11 5468 2606
12 5989 2873
13 4706 2674
14 4612 2035
15 6347 2683
16 6107 669
17 7611 5184
18 7462 3590
19 7732 4723
20 5900 3561
21 4483 3369
22 6101 1110
23 5199 2182
24 1633 2809
25 4307 2322
26 675 1006
27 7555 4819
28 7541 3981
29 3177 756
30 7352 4506
31 7545 2801
32 3245 3305
33 6426 3173
34 4608 1198
35 23 2216
36 7248 3779
37 7762 4595
38 7392 2244
39 3484 2829
40 6271 2135
41 4985 140
42 1916 1569
43 7280 4899
44 7509 3239
45 10 2676
46 6807 2993
47 5185 3258
48 3023 1942
`;

// Parse the cities data into an array of objects
const cities = citiesData.trim().split('\n').map(line => {
  const [id, x, y] = line.trim().split(/\s+/).map(Number);
  return { id, x, y };
});

// Calculate Euclidean distance between two cities
function calculateDistance(city1, city2) {
  const dx = city1.x - city2.x;
  const dy = city1.y - city2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// Calculate the total distance of a tour
function calculateTourDistance(tour, cities) {
  let totalDistance = 0;
  for (let i = 0; i < tour.length - 1; i++) {
    const city1 = cities[tour[i] - 1];
    const city2 = cities[tour[i + 1] - 1];
    totalDistance += calculateDistance(city1, city2);
  }
  // Add distance from last city back to first city to complete the tour
  const firstCity = cities[tour[0] - 1];
  const lastCity = cities[tour[tour.length - 1] - 1];
  totalDistance += calculateDistance(lastCity, firstCity);
  return totalDistance;
}

// Implementation of 2-opt algorithm for TSP
function twoOptImprovement(tour, cities) {
  let improved = true;
  let bestDistance = calculateTourDistance(tour, cities);
  
  while (improved) {
    improved = false;
    
    for (let i = 1; i < tour.length - 1; i++) {
      for (let j = i + 1; j < tour.length; j++) {
        // Skip adjacent edges
        if (j - i === 1) continue;
        
        // Create a new tour by reversing the segment between i and j
        const newTour = [...tour];
        // Reverse the segment between i and j
        let left = i;
        let right = j;
        while (left < right) {
          [newTour[left], newTour[right]] = [newTour[right], newTour[left]];
          left++;
          right--;
        }
        
        const newDistance = calculateTourDistance(newTour, cities);
        
        if (newDistance < bestDistance) {
          tour = newTour;
          bestDistance = newDistance;
          improved = true;
          // Break out of the inner loop to restart with the new tour
          break;
        }
      }
      if (improved) break;
    }
  }
  
  return { tour, distance: bestDistance };
}

// Generate an initial tour using nearest neighbor heuristic
function nearestNeighborTour(cities) {
  const n = cities.length;
  const tour = [1]; // Start with city 1
  const unvisited = new Set(cities.map(city => city.id).filter(id => id !== 1));
  
  let currentCityId = 1;
  
  while (unvisited.size > 0) {
    const currentCity = cities[currentCityId - 1];
    let nearestCityId = null;
    let minDistance = Infinity;
    
    // Find the nearest unvisited city
    for (const nextCityId of unvisited) {
      const nextCity = cities[nextCityId - 1];
      const distance = calculateDistance(currentCity, nextCity);
      
      if (distance < minDistance) {
        minDistance = distance;
        nearestCityId = nextCityId;
      }
    }
    
    tour.push(nearestCityId);
    unvisited.delete(nearestCityId);
    currentCityId = nearestCityId;
  }
  
  return tour;
}

// Simulated annealing for TSP
function simulatedAnnealing(cities, initialTemperature = 100, coolingRate = 0.995, stoppingTemperature = 0.01) {
  // Start with a random tour
  let tour = Array.from({length: cities.length}, (_, i) => i + 1);
  
  // Fisher-Yates shuffle to randomize the initial tour
  for (let i = tour.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [tour[i], tour[j]] = [tour[j], tour[i]];
  }
  
  let currentDistance = calculateTourDistance(tour, cities);
  let bestTour = [...tour];
  let bestDistance = currentDistance;
  
  let temperature = initialTemperature;
  
  // Main simulated annealing loop
  while (temperature > stoppingTemperature) {
    // Create a new solution by swapping two cities
    const newTour = [...tour];
    const i = 1 + Math.floor(Math.random() * (tour.length - 1));
    const j = 1 + Math.floor(Math.random() * (tour.length - 1));
    
    [newTour[i], newTour[j]] = [newTour[j], newTour[i]];
    
    const newDistance = calculateTourDistance(newTour, cities);
    
    // Decide whether to accept the new solution
    if (newDistance < currentDistance || Math.random() < Math.exp((currentDistance - newDistance) / temperature)) {
      tour = newTour;
      currentDistance = newDistance;
      
      if (newDistance < bestDistance) {
        bestTour = [...tour];
        bestDistance = newDistance;
      }
    }
    
    // Cool down
    temperature *= coolingRate;
  }
  
  // Apply 2-opt improvement to the best tour found
  const finalImprovement = twoOptImprovement(bestTour, cities);
  
  return finalImprovement;
}

// Hybrid approach combining multiple methods
function hybridTSPSolver(cities) {
  // Run nearest neighbor
  const nnTour = nearestNeighborTour(cities);
  const nnSolution = { tour: nnTour, distance: calculateTourDistance(nnTour, cities) };
  
  // Run several simulated annealing attempts with different parameters
  const saSolution1 = simulatedAnnealing(cities, 100, 0.99, 0.1);
  const saSolution2 = simulatedAnnealing(cities, 200, 0.995, 0.01);
  
  // Choose the best solution found
  const solutions = [nnSolution, saSolution1, saSolution2];
  solutions.sort((a, b) => a.distance - b.distance);
  
  const bestSolution = solutions[0];
  
  // Apply final 2-opt improvement
  return twoOptImprovement(bestSolution.tour, cities);
}

// Run the solvers
const nnSolution = twoOptImprovement(nearestNeighborTour(cities), cities);
const saSolution = simulatedAnnealing(cities);
const hybridSolution = hybridTSPSolver(cities);

// Find the best solution among all approaches
const allSolutions = [nnSolution, saSolution, hybridSolution];
allSolutions.sort((a, b) => a.distance - b.distance);
const bestOverallSolution = allSolutions[0];

console.log(`Best Tour: ${bestOverallSolution.tour.join(' -> ')} -> ${bestOverallSolution.tour[0]}`);
console.log(`Total distance: ${bestOverallSolution.distance.toFixed(2)}`);
```
Optimal Tour:
```
1 → 8 → 9 → 38 → 31 → 44 → 18 → 7 → 28 → 6 → 19 → 37 → 27 → 17 → 43 → 30 → 36 → 46 → 33 → 15 → 12 → 20 → 47 → 21 → 39 → 32 → 24 → 45 → 35 → 4 → 26 → 10 → 42 → 2 → 29 → 5 → 48 → 25 → 14 → 13 → 11 → 23 → 34 → 41 → 16 → 22 → 3 → 40 → 1
```
Optimal Tour Distance
```
34,626.71
```
### Reward Function
- Accuracy: How close is output optimal tour distance to pre-computed reference optimal tour distance? Reward closer to 0 the higher policy model output is/further away from reference optimal tour distance
- Format: Do we want to encourage and reward \<think>\</think> reasoning process followed by \<answer>\</answer> code solution output format.
## Minimal Pipeline Code Sample - Inference
- dataset -> problem -> input -> some model -> extract output -> pass to reward function
- Code that takes problem from dataset, forms model input, extracts output, runs output, and passes to reward function (no grpo implementation, just inference pipeline)
## Choosing A Model
- model family
    - llama
    - qwen
    - mistral
    - r1 distilled versions
    - ideally want model optimized for mathematical/geometric reasoning and translating to code solution
- model size
    - tradeoff between using more powerful starter model but having to use more aggresive memory optimization techniques
    - will likely have to use Parameter enhanced fine tuning (LORA)
    - if use larger model we will likely have to quantize model weights and use PEFT (QLORA)
## Minimal GRPO Code Sample
- simple grpo implemenation optimizing for simpler problem (GSM8K)
## Helpful GRPO Readings
- [Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/)
- [Minimal GRPO Code Sample](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb?ref=ghost.oxen.ai)
- [GRPO VRAM Requirements For the GPU Poor](https://ghost.oxen.ai/grpo-vram-requirements-for-the-gpu-poor/)
- [How DeepSeek R1, GRPO, and Previous DeepSeek Models Work](https://www.oxen.ai/blog/how-deepseek-r1-grpo-and-previous-deepseek-models-work?ref=ghost.oxen.ai)
## Random Thoughts
- solving tsp for 30 + cities requires composing multiple heuristic solutions -> model output could become too large / too memory intensive (can maybe provide implementations of heuristics that the model is likely to use -> therefore model only needs to output reasoning trace and then a suggested set of heuristics to attempt/compose -> then we run that, retrieve optimal tour distance and pass to reward function)
- system prompt for training: (encourages answer format) 
![](image.png)