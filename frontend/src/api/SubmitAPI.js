import Setting from '../config';

class SubmitAPI {
  static submit(api, body) {
    return fetch(`${Setting.backEndUrl}/${api}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });
  }
}

export default SubmitAPI;
