from .agent import run_agent

#run this file in terminal
if __name__ == "__main__":
    while True:
        q = input("you> ")
        if q.strip() in {"exit", "quit"}:
            break
        print(run_agent(q))