
import rpyc

class Csci378GraderService(rpyc.Service):
    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_submit_response(self, question_id, response): # this is an exposed method
        return self.private_submit_response(question_id, response)

    exposed_the_real_answer_though = 43     # an exposed attribute

    private_answers = {
                'q1': 'olduvai gorge',
                'q2': 'museum'
            }

    private_replies = {
                'q1': "Correct! Proceed to https://markandrewhopkins.com",
                'q2': ' '.join("""Correct! You donate the jewel to the British Museum, 
                where it can go on display for (some of) the world to enjoy.
                Congratulations! Project 1 is complete.""".split())
            }

    def private_submit_response(self, question_id, response):  # while this method is not exposed
        response = response.strip().lower()
        answers = Csci378GraderService.private_answers
        replies = Csci378GraderService.private_replies
        if question_id not in answers:
            return "Question id {} is not a valid id.".format(question_id)
        if response == answers[question_id]:
            return replies[question_id]
        else:
            return "Incorrect response! Try again."
    

if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(Csci378GraderService, port=18861)
    t.start()