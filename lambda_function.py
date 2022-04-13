# -*- coding: utf-8 -*-

# This sample demonstrates handling intents from an Alexa skill using the Alexa Skills Kit SDK for Python.
# Please visit https://alexa.design/cookbook for additional examples on implementing slots, dialog management,
# session persistence, api calls, and more.
# This sample is built using the handler classes approach in skill builder.
import logging
import ask_sdk_core.utils as ask_utils

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Welcome, you can say Hello or Help. Which would you like to try?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

class symptoms_present(AbstractRequestHandler):
    
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("symptoms_present")(handler_input)
        
    def handle(self, handler_input):
        slots = handler_input.request_envelope.request.intent.slots
        symptomstype = slots["symptomstype"].value
        
        if symptomstype == "rashes":
            speak_output = "The symptom seem to be of Fungal Infection"
        elif symptomstype == "itching":
            speak_output = "It is one of the symptom of Fungal Infection"
        elif symptomstype == "cold":
            speak_output = "It is one of the symptom of Flu"
        elif symptomstype == "fever":
            speak_output = "As per your symptoms it seems you might have Flu"
        elif symptomstype == "headache":
            speak_output = "It is one of the symptom of Flu"
        elif symptomstype == "soar_throat":
            speak_output = "It is one of the symptom of Flu"  
        elif symptomstype == "runny_nose":
            speak_output = "It is one of the symptom of Flu"
        elif symptomstype == "stomach_ache":
            speak_output = "The symptom seem to be of either Gastroenteritis or Diarrhea"   
        elif symptomstype == "vomit":
            speak_output = "It is one of the symptom of Gastroenteritis as well as Jaundice"
        elif symptomstype == "cramps":
            speak_output = "It is one of the symptom of Diarrhea"
        elif symptomstype == "swelling_on_stomach":
            speak_output = "As per your symptoms it seems you might have jaundice"   
        elif symptomstype == "spot_urination":
            speak_output = "It is one of the symptom of Urinary Track Infection"
        elif symptomstype == "burning_in_urinary_track":
            speak_output = "It is one of the symptom of Urinary Track Infection"
	    else:
            speak_output = "I am not sure what you mean!"
        
        return (
            handler_input.response_builder.speak(speak_output).response
            )

class doctors(AbstractRequestHandler):
    
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("doctors")(handler_input)
        
    def handle(self, handler_input):
        slots = handler_input.request_envelope.request.intent.slots
        diseasetype = slots["diseasetype"].value
        
        if diseasetype == "Fungal_Infection":
            speak_output = "Contact Dr Gayatri Bala Juneja, she is a specialist"
        elif symptomstype == "Flu":
            speak_output = "Contact Dr Nipun Jain to get the best advice"
        elif symptomstype == "Gastroenteritis":
            speak_output = "Contact Dr SK Kashyap"
        elif symptomstype == "Diarrhea":
            speak_output = "Contact Dr Jayant Shastri"
        elif symptomstype == "Jaundice":
            speak_output = "Contact Dr Suman Mohan, she is a specialist"
        elif symptomstype == "Urinary_track_infection":
            speak_output = "Contact Dr Anirban Biswas, he has been listed as one of the top gynaecologist"  
	    else:
            speak_output = "Sorry, I am not sure about this!"
        
        return (
            handler_input.response_builder.speak(speak_output).response
            )

class ConfirmAppointmentIntent(AbstractRequestHandler):
    
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("ConfirmAppointmentIntent")(handler_input)
        
    def handle(self, handler_input):
        speak_output = "Got it. Your appointment has been scheduled."
        
        return (
            handler_input.response_builder.speak(speak_output).response
            )


class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "You can say hello to me! How can I help?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Goodbye!"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )

class FallbackIntentHandler(AbstractRequestHandler):
    """Single handler for Fallback Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        logger.info("In FallbackIntentHandler")
        speech = "Hmm, I'm not sure. You can say Hello or Help. What would you like to do?"
        reprompt = "I didn't catch that. What can I help you with?"

        return handler_input.response_builder.speak(speech).ask(reprompt).response

class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # Any cleanup logic goes here.

        return handler_input.response_builder.response


class IntentReflectorHandler(AbstractRequestHandler):
    """The intent reflector is used for interaction model testing and debugging.
    It will simply repeat the intent the user said. You can create custom handlers
    for your intents by defining them above, then also adding them to the request
    handler chain below.
    """
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("IntentRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        intent_name = ask_utils.get_intent_name(handler_input)
        speak_output = "You just triggered " + intent_name + "."

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors. If you receive an error
    stating the request handler chain is not found, you have not implemented a handler for
    the intent being invoked or included it in the skill builder below.
    """
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Sorry, I had trouble doing what you asked. Please try again."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

# The SkillBuilder object acts as the entry point for your skill, routing all request and response
# payloads to the handlers above. Make sure any new handlers or interceptors you've
# defined are included below. The order matters - they're processed top to bottom.


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(symptoms_present())
sb.add_request_handler(doctors())
sb.add_request_handler(ConfirmAppointmentIntent())
sb.add_request_handler(HelloWorldIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(IntentReflectorHandler()) # make sure IntentReflectorHandler is last so it doesn't override your custom intent handlers

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
