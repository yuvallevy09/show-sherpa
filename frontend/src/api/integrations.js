// NOTE: Previous vendor integrations were removed. This module is kept as a compatibility shim
// in case something still imports it, but all functions are currently unimplemented.

function notImplemented(name) {
  return async () => {
    throw new Error(`${name} is not implemented. Replace integrations with backend endpoints.`);
  };
}

export const Core = {};

export const InvokeLLM = notImplemented("InvokeLLM");
export const SendEmail = notImplemented("SendEmail");
export const UploadFile = notImplemented("UploadFile");
export const GenerateImage = notImplemented("GenerateImage");
export const ExtractDataFromUploadedFile = notImplemented("ExtractDataFromUploadedFile");
export const CreateFileSignedUrl = notImplemented("CreateFileSignedUrl");
export const UploadPrivateFile = notImplemented("UploadPrivateFile");






