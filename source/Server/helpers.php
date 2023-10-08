<?php
    const MAX_MINUTES_FOR_INACTIVITY = 60; 

    function connectToDB()
    {
        $db = new mysqli("localhost", "[user name]", "[password]", "[database]");
        if ($db->connect_error) {
            exit("Connection error: " . mysqli_connect_error());
        }
        return $db;
    }

    function checkPw($db, $userName, $pw)
    {
        $success = false;

        if ($response = $db->query(
            "SELECT 
                u.PW_HASH as pwHash,
                u.SALT as salt,
                u.ACTIVATION_CODE as activationCode
            FROM user u
            WHERE u.NAME='".addslashes($userName)."'")) {

            $obj = $response->fetch_object();
            if (!$obj) {
                return false;
            }
            $pwHash = hash("sha256", $pw . $obj->salt);

            return $pwHash == $obj->pwHash && $obj->activationCode == "";
        }
        return false;
    }

    function checkPwAndActivationCode($db, $userName, $pw, $activationCode)
    {
        $success = false;

        if ($response = $db->query(
            "SELECT 
                u.PW_HASH as pwHash,
                u.SALT as salt,
                u.ACTIVATION_CODE as activationCode
            FROM user u
            WHERE u.NAME='".addslashes($userName)."'")) {

            $obj = $response->fetch_object();
            if (!$obj) {
                return false;
            }
            $pwHash = hash("sha256", $pw . $obj->salt);

            return $pwHash == $obj->pwHash && $obj->activationCode == $activationCode;
        }
        return false;
    }

    function sendDiscordMessage($payload) {
        if (strlen($payload) >= 0) {
            $ch = curl_init("[webhook URL]");
            curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-type: application/json'));
            curl_setopt($ch, CURLOPT_POST, 1);
            curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
            curl_setopt($ch, CURLOPT_FOLLOWLOCATION, 1);
            curl_setopt($ch, CURLOPT_HEADER, 0);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
            $response = curl_exec($ch);
            curl_close($ch);
            return $response;
        }
    }

?>