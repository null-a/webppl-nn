'use strict';

module.exports = function(env) {

  function inOptimize(s, k, a) {
    return k(s, env.coroutine.paramsSeen !== undefined);
  }

  function inGuide(s, k, a) {
    return k(s, env.coroutine._guide);
  }

  return {
    inOptimize: inOptimize,
    inGuide: inGuide
  };

};
