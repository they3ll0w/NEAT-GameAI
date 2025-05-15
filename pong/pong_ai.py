import os
import pickle
import pygame
import neat
import argparse
from pong_game import Game


class PongGame:
    """
    Class to handle the NEAT AI training and testing for the Pong game.
    """
    def __init__(self, window, width, height):
        """
        Initialize the PongGame with a game instance and references to game objects.
        
        Args:
            window: Pygame window surface
            width: Width of the game window
            height: Height of the game window
        """
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, genome, config):
        """
        Test the AI against a human player.
        
        Args:
            genome: Trained NEAT genome
            config: NEAT configuration
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        clock = pygame.time.Clock()

        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            # Human player controls
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            # AI controls
            output = net.activate((
                self.right_paddle.y, 
                self.ball.y, 
                abs(self.right_paddle.x - self.ball.x), 
                self.ball.x_vel, 
                self.ball.y_vel
            ))
            decision = output.index(max(output))
            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            # Update game state
            game_info = self.game.loop()
            self.game.draw()
            pygame.display.update()

        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        """
        Train two AI instances against each other.
        
        Args:
            genome1: First NEAT genome
            genome2: Second NEAT genome
            config: NEAT configuration
        """
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        max_hits = 50 
        
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Left paddle AI
            output1 = net1.activate((
                self.left_paddle.y, 
                self.ball.y, 
                abs(self.left_paddle.x - self.ball.x), 
                self.ball.x_vel, 
                self.ball.y_vel
            ))
            decision1 = output1.index(max(output1))

            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            # Right paddle AI
            output2 = net2.activate((
                self.right_paddle.y, 
                self.ball.y, 
                abs(self.right_paddle.x - self.ball.x), 
                self.ball.x_vel, 
                self.ball.y_vel
            ))
            decision2 = output2.index(max(output2))

            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            # End game conditions
            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > max_hits:
                self.calculate_fitness(genome1, genome2, game_info)
                break

    def calculate_fitness(self, genome1, genome2, game_info):
        """
        Calculate fitness scores for both genomes based on game performance.
        
        Args:
            genome1: First NEAT genome
            genome2: Second NEAT genome
            game_info: Information about the game state
        """
        # Base fitness on successful hits
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits
        
        if game_info.left_score > 0:
            genome1.fitness += 5
        if game_info.right_score > 0:
            genome2.fitness += 5


def eval_genomes(genomes, config):
    """
    Evaluate genomes by having them play against each other.
    
    Args:
        genomes: List of NEAT genomes to evaluate
        config: NEAT configuration
    """
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    for i, (genome_id1, genome1) in enumerate(genomes):

        if i == len(genomes)-1: break

        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness is None else genome2.fitness

            game = PongGame(window, width, height)
            game.train_ai(genome1, genome2, config)


def run_neat(config, checkpoint=None, generations=50):
    """
    Run the NEAT algorithm to train the AI.
    
    Args:
        config: NEAT configuration
        checkpoint: Optional checkpoint file to restore from
        generations: Number of generations to train for
    
    Returns:
        The best genome from training
    """

    # Create population from checkpoint or config
    if checkpoint and os.path.exists(checkpoint):
        print(f"Restoring from checkpoint: {checkpoint}")
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("Starting a new population")
        p = neat.Population(config)

    # Add reporters to track progress
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    # Run the algorithm
    winner = p.run(eval_genomes, generations)
    
    # Save the winner
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
    
    return winner


def test_ai(config):
    """
    Test the best AI against a human player.
    
    Args:
        config: NEAT configuration
    """
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)

    game = PongGame(window, width, height)
    game.test_ai(winner, config)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NEAT Pong AI')
    parser.add_argument('--test', action='store_true', help='Test the AI')
    parser.add_argument('--train', action='store_true', help='Train the AI')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file to restore from')
    parser.add_argument('--generations', type=int, default=30, help='Number of generations to train for')
    args = parser.parse_args()

    # Set up configuration
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Run training or testing based on arguments
    if args.test:
        test_ai(config)
    elif args.train:
        run_neat(config, args.checkpoint, args.generations)
    else:
        # Default behavior - train then test
        winner = run_neat(config, args.checkpoint, args.generations)
        test_ai(config)
