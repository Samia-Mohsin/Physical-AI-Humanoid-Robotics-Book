import React from 'react';
import { UserProvider } from './contexts/UserContext';
import { LanguageProvider } from './contexts/LanguageContext';

// This is the main App wrapper that ensures contexts are available for all pages
const App = ({ children }) => {
  return (
    <UserProvider>
      <LanguageProvider>
        {children}
      </LanguageProvider>
    </UserProvider>
  );
};

export default App;