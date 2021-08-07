import os
from twilio.rest import Client
from .models import (Patient, Message, UserData)
from dateutil.parser import parse

def get_prepared_string(s):
    return s.casefold().strip()

def is_yes_or_no(s):
    return get_prepared_string(s) in ['yes', 'no']

def parse_date(s):
    try:
        parse(s)
        return True
    except ValueError:
        return False

class RecieveSend:
    def __init__(self):
        #self.full_reset()
        #self.clear_messages()

        values = open("/home/blueuser/PycharmProjects/virus-valet/secrets.hidden", "r").read().split()
        self.sid = values[0]
        self.token = values[1]
        self.client = Client(self.sid, self.token)
        self.default_number = "+12262708145"
        self.questions = {
            "What is your address?" :
                lambda answer, patient: get_prepared_string(answer) == get_prepared_string(patient.address),
            "What is your date of birth?" :
                lambda answer, patient: get_prepared_string(answer) == get_prepared_string(str(patient.date_of_birth))
            ,
            "Have you suffered from any of the following symptoms in the past 14 days?\nFever or chills\nCough\nShortness of breath or difficulty breathing\nFatigue\nMuscle or body aches\nHeadache\nNew loss of taste or smell\nSore throat\nCongestion or runny nose\nNausea or vomiting\nDiarrhea":
                lambda answer, patient: is_yes_or_no(answer),
            "Where could you have acquired your infection in the past 14 days?\nTraveling to a different country\nAttending a party\netc.":
                lambda answer, patient: True,
            "Do you attend school or a workplace?":
                lambda answer, patient: is_yes_or_no(answer),
            "If yeas, What is your school or workplaces name?":
                lambda answer, patient: True,
            "Have you started self-isolating?":
                lambda answer, patient: is_yes_or_no(answer),
            "If so, when did you start self-isolating?":
                lambda answer, patient: parse_date(answer),
        }

    def full_reset(self):
       for i in Patient.objects.all():
           i.delete()
       for i in Message.objects.all():
           i.delete()

    def clear_messages(self):
        for i in Message.objects.all():
            i.delete()
        for i in UserData.objects.all():
            i.delete()

    def print_messages(self):
        for i in Message.objects.all():
            print(i, i.id)

    def send_message(self, body, number_to, number_from = None):
        if number_from is None:
            number_from = self.default_number
        message = self.client.messages \
            .create(
            body=body,
            from_=number_from,
            to=number_to
        )

    def check_answer(self, patient, body):
        messages = Message.objects.filter(patient = patient)
        questions = [i for i in messages if i.is_question]
        if len(questions):
            questions.reverse()
            for current_question in questions:
                checking_function = self.questions.get(current_question.message)
                if checking_function and checking_function(body, patient):
                    return current_question.id
            return -1
        return -1


    def save_messages(self, request, is_patient, is_nurse = False, is_question = False, real_request = True):
        if real_request:
            pull = lambda x : request.POST.get(x)
        else:
            pull = lambda x: request.get(x)
        body = pull("Body")
        phone_number = pull("From")

        print()
        print("Saved:", body, phone_number)
        print()
        try:
            patient = Patient.objects.filter(phone_number = phone_number)[0]
        except:
            print("patients:", Patient.objects.filter(phone_number = phone_number))
        messages = list(Message.objects.filter(patient = patient))
        nurse_questions = [i for i in messages if i.is_question and i.sent_by_nurse]


        answer_ids = [i.is_answer for i in Message.objects.filter(patient = patient) if i.is_answer > 0]
        is_answer = self.check_answer(patient, body) # and is_patient and not is_question
        if not is_nurse and is_answer in answer_ids:
            is_answer = -1
        if is_nurse:
            is_answer = -1
        if len(messages) and len(nurse_questions) and messages[-1].id == nurse_questions[-1].id:
            is_answer = nurse_questions[-1].id
            print("this is a nurse answer")
        message = Message(patient = patient, message = body, is_patient = is_patient, is_question = is_question, is_answer = is_answer, sent_by_nurse = is_nurse)
        message.save()
        return is_answer


    def send_questions(self, patient, is_answer = 1):
        messages = Message.objects.filter(patient = patient)
        bot_questions = [i.message for i in messages if i.is_question and not i.sent_by_nurse]
        nurse_questions = [i.message for i in messages if i.is_question and i.sent_by_nurse]
        questions = [i.message for i in messages if i.is_question]
        if len(messages) and len(bot_questions) and is_answer < 0 and bot_questions[-1] != list(self.questions.keys())[-1]:
            self.send_message(f"Sorry, but that answer wasn't recognized.\n\n{bot_questions[-1]}", str(patient.phone_number))
            self.save_messages({"Body": f"Sorry, but that answer wasn't recognized.\n\n{questions[-1]}", "From": str(patient.phone_number)}, is_patient=False, real_request=False)
            return
        question_ids = [i.id for i in messages if i.is_question]
        answer_ids = [i.is_answer for i in messages if i.is_answer > 0]
        if len(question_ids) != len(answer_ids):
            return
        for i in self.questions:
            if i not in questions:
                self.send_message(i, str(patient.phone_number))
                self.save_messages({"Body": i, "From": str(patient.phone_number)}, is_patient=False, is_question=True, real_request=False)
                return
        return

    def gather_user_data(self, patient):
        data = UserData.objects.filter(patient = patient)
        if len(data) and False:
            print(data[0].is_symptomatic, data[0].attending_public, data[0].not_isolating)
            return data[0]
        else:
            print("doing")
            print(patient)
            print(patient.phone_number)
            messages = Message.objects.filter(patient = patient)
            print(messages)
            if not len(messages):
                print(len(messages))
                data = UserData(patient=patient)
                data.save()
                return data
            question_answers = {Message.objects.get(id = i.is_answer).message: i.message for i in messages if i.is_answer > 0}
            def find_responses_related(x, check_list):
                try:
                    values = [question_answers.get(i) for i in question_answers if x in i][0].lower()
                    return values in check_list
                except:
                    return False

            is_symptomatic = find_responses_related("symptoms in the past 14 days", ['yes', 'y'])
            attending_public = find_responses_related("Where could you have acquired your infection in", ['yes', 'y'])
            not_isolating = find_responses_related("Have you started self-isolating", ["n", "no"])
            print("symptoms")
            print(is_symptomatic, attending_public, not_isolating)
            data = UserData(patient = patient, is_symptomatic = is_symptomatic, attending_public = attending_public, not_isolating = not_isolating)
            data.save()
            return data





            '''   for patient in patients:
        if patient.is_symptomatic and (patient.attended_school_or_workplace or patient.not_isolating):
            patient["img"] = img_map["red"]
        elif patient.is_symptomatic or patient.attended_school_or_workplace or patient.not_isolating:
            patient["img"] = img_map["yellow"]
        else:
            patient["img"] = img_map["green"]'''


