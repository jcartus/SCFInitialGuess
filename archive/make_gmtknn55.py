from SCFInitialGuess.generate_dataset import main as program


def main():
    
    program(
        source="database/GMTKN55",
        destination="dataset/GMTKN55",
        amplification=0,
        number_of_processes=8
    )

if __name__ == '__main__':
    main()
